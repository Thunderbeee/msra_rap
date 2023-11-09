from typing import List, Optional
import fire
from llama_new import Llama
from tqdm import tqdm
from rap.utils.gsm8k import judge_answer_gsm8k, get_gsm8k_dataset
from rap.utils.multiarith import judge_answer_multiarith, get_multiarith_dataset
import re

few_shots = """
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n
Answer: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\n
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n
Answer: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\n
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n
Answer: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\n
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n
Answer: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\n
Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n
Answer: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\n
Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n
Answer: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\n
Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n
Answer: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33\n\n
"""

def main(
    ckpt_dir: str = '/home/xyyue/zangwei/mingyuan/llama/llama-2-7b-chat',
    tokenizer_path: str = '/home/xyyue/zangwei/mingyuan/llama/llama-2-7b/tokenizer.model',
    self_consistency_number = 10,
    dataset = 'gsm8k',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if dataset == 'gsm8k':
        examples = get_gsm8k_dataset('test')
    elif dataset == 'multiarith':
        examples = get_multiarith_dataset('test')
    else:
        raise NotImplementedError()

    total_correct = 0
    for i, example in enumerate((pbar := tqdm(examples, position=1))):
        question = example['question']
        answer = example['answer']
    
        if dataset == 'gsm8k':
            answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
            answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')
        
        prompt = "Answer the following question. Please answer it in a complete sentence, ending with \"The answer is\"" + question
        input_prompt = few_shots + prompt
        voting = {}
        for _ in range(self_consistency_number):
            dialogs = [[{"role":"user", "content": input_prompt}]]
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            output, _ = judge_answer_gsm8k(results[0]['generation']['content'], answer)
            voting[output] = voting.get(output, 0) + 1

        max_item = max(voting, key=voting.get)
        total_correct += max_item == answer
        pbar.set_description(f'{total_correct} / {i+1} = {total_correct/(i+1):.2f}')


if __name__ == "__main__":
    fire.Fire(main)
