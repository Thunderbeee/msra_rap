import pickle
import re
from datetime import datetime
from rap.models import QueryLlama
from rap.utils.gsm8k import judge_answer_gsm8k, get_gsm8k_dataset
from rap.gsm8k_mcts import reasoning_mcts_search
from rap.helpers import *
from typing import Tuple
import os
import sys
import torch
import torch.distributed
import torch.backends.cudnn
import fire
import time
import json
import random
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, LLaMA 
from transformers import AutoTokenizer, LlamaForCausalLM


def setup_logging(log_dir, llama_ckpt):
    if log_dir is None:
        log_dir = f'logs/gsm8k_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)   
    return log_dir


def setup_random():
    # set random seed 
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, max_batch_size: int) -> LLaMA:
    start_time = time.time()
    print(ckpt_dir)
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    print(checkpoints)
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    
    #! huggingface version of loading LLAMA =======================
    # tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/llama/llama-2-7b-chat-to-hf/")
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    # model = LlamaForCausalLM.from_pretrained("/root/autodl-fs/llama/llama-2-7b-chat-to-hf/").half().cuda().eval() #! add "half()" to fit in a smaller GPU
    #! ============================================================
    
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main_mcts(llama_ckpt='/data/luoyingtao/llama/llama-2-13b',
              decompose_examples='data/gsm8k/prompts/decompose_examples.json',
              useful_examples='data/gsm8k/prompts/useful_examples.json',
              max_batch_size=2,
              max_response_length=200,
              mcts_rollouts=10,
              n_sample_subquestion=4,
              n_sample_confidence=8,
              temperature=0.8,
              max_depth=6,
              w_exp=1,
              r_alpha=0.5,
              r1_default=1,
              resume=0,
              log_dir=None,
              speedup_confidence_batch_size=None):
    setup_random()
    log_dir = setup_logging(log_dir, llama_ckpt)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    tokenizer_path = os.path.join(llama_ckpt, "tokenizer.model")
    llama = load(llama_ckpt, tokenizer_path, local_rank, world_size, max_batch_size)
    world_model = QueryLlama(llama, max_response_length=max_response_length, log_file=None)

    examples = get_gsm8k_dataset('test')
    with open(decompose_examples) as f:
        decompose_examples = json.load(f)
    """ 
    {
        "input": 
            "Given a question, please decompose it into sub-questions. 
            For each sub-question, please answer it in a complete sentence, ending with \"The answer is\". 
            When the original question is answerable, please start the subquestion with \"Now we can answer the question: \".\n\n
            Question 1: ...\n
                Question 1.1: ...\nAnswer 1.1: ...\n
                Question 1.2: ...\nAnswer 1.2: ...\n
                Question 1.3: ...\nAnswer 1.3: ...\n
                Question 1.4: ...\nAnswer 1.4: ...\n\n
            Question 2: ...\n
                Question 2.1: ...\nAnswer 2.1: ...\n
                Question 2.2: ...\nAnswer 2.2: ...\n\n
            Question 3: ...\n
                Question 3.1: ...\nAnswer 3.1: ...\n
                Question 3.2: ...\nAnswer 3.2: ...\n
                Question 3.3: ...\nAnswer 3.3: ...\n
                Question 3.4: ...\nAnswer 3.4: ...\n
                Question 3.5: ...\nAnswer 3.5: ...\n\n
            Question 4: ...\n
                Question 4.1: ...\nAnswer 4.1: ...\n
                Question 4.2: ...\nAnswer 4.2: ...\n\n",
        "question_prefix": "Question 5: ",
        "subquestion_prefix": "Question 5.{}:",
        "overall_question_prefix": "Question 5.{}: Now we can answer the question: {}\n",
        "answer_prefix": "Answer 5.{}:",
        "index": 5
    }
    """
    with open(useful_examples) as f:
        useful_examples = json.load(f)
    """ 
    {
        "input": 
            "Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. 
            Output 'Yes' or 'No', and a reason.\n\n
            Question 1: ...\n
                Question 1.1: ...\n
                Question 1.2: ...\n
                New question 1.3: ...\n
                Is the new question useful? Yes. We need the answer to calculate how old is Kody now.\n\n
            Question 2: ...\n
                New question 2.1: ...\n
                Is the new question useful? No. The new question is not related to the original question.\n\n
            Question 3: ...\n
                Question 3.1: ...\n
                Question 3.2: ...\n
                New question 3.3: ...\n
                Is the new question useful? Yes. We need the answer to calculate the total construction costs.\n\n
            Question 4: ...\n
                Question 4.1: ...\n
                New question 4.2: ...\n
                Is the new question useful? No. It is too hard to answer the new question based on the current information.\n\n",
        "question_prefix": "Question 5: ",
        "subquestion_prefix": "Question 5.{}:",
        "new_subquestion_prefix": "New question 5.{}:",
        "answer_prefix": "Is the new question useful?"
    }
    """ 

    total_correct = [0] * mcts_rollouts 
    for i, example in enumerate((pbar := tqdm(examples, disable=local_rank > 0, position=1))):
        if i < resume:
            continue
        question = example['question']
        answer = example['answer']
        answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
        answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')
        
        # max_depth = determine_max_depth(world_model, question)
        # print(f"==> the model thinks the max depth should be {max_depth}")
        # continue
        
        #! ========================================
        trajs, tree, trees, extra_info = reasoning_mcts_search(question, decompose_examples, useful_examples, world_model,
                                                   n_sample_subquestion=n_sample_subquestion,
                                                   mcts_rollouts=mcts_rollouts,
                                                   n_sample_confidence=n_sample_confidence,
                                                   temperature=temperature,
                                                   max_depth=max_depth,
                                                   w_exp=w_exp,
                                                   r_alpha=r_alpha,
                                                   r1_default=r1_default,
                                                   eos_token_id=world_model.tokenizer.encode('\n', bos=False, eos=False)[-1],
                                                   speedup_confidence_batch_size=speedup_confidence_batch_size)
        #! ========================================
        
        if local_rank == 0:
            json_logs = []
            for rollout, traj in enumerate(trajs):
                output, correct = judge_answer_gsm8k(traj, answer)
                json_logs.append({
                    'rollout': rollout + 1,
                    'question': question,
                    'answer': answer,
                    'output': output,
                    'correct': correct,
                    'traj': traj,
                    'query_LM_counter': extra_info.query_LM_counter,
                    'num_hit_max_depth': extra_info.num_hit_max_depth,
                    'exec_time': extra_info.exec_time
                })
                total_correct[rollout] += correct
            with open(os.path.join(log_dir, f'{i:04d}.json'), 'w') as f:
                json.dump(json_logs, f, indent=2)
            with open(os.path.join(log_dir, f'{i:04d}.tree'), 'w') as f:
                f.write(tree)
            with open(os.path.join(log_dir, f'{i:04d}.pkl'), 'wb') as f:
                pickle.dump(trees, f)
            tqdm.write(' '.join(f'{c/(i+1-resume):0.3f}' for c in total_correct))
            pbar.set_description(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')


if __name__ == '__main__':
    fire.Fire(main_mcts)