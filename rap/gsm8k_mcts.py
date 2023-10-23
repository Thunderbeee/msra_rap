import io
import os
import random
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm, trange

from .mcts import MCTS, MCTSNode
from .models import QueryLM


def reach_terminal_subquestion(partial_solution, question_group_id):
    generated_question_group = partial_solution.split('\n\n')[-1]
    if 'Now we can answer' in generated_question_group:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True
    if f'Question {question_group_id}.' not in generated_question_group:
        return False
    
    last_subquestion = generated_question_group.split(f'Question {question_group_id}.')[-1].split('\n')[0]
    undecomposed_question = generated_question_group.split('\n')[0]
    if last_subquestion.lower() in undecomposed_question.lower():
        return True
    return False


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, decompose, useful, gen_fn, reward_fn, depth, r1_default, r_alpha, prompt_index,
                 parent: 'ReasoningMCTSNode' = None, r0=0.):
        self._conf = None
        self.children = []
        
        self.partial_solution = decompose    #! state 
        self.useful = useful
        
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self._r0 = r0
        self._r1 = self._r1_default = r1_default
        self._r_alpha = r_alpha
        self._ans_list = None
        self._visited = False
        self.parent = parent
        self._prompt_index = prompt_index

    def _child_node(self, prompt, question_prompt, r0):
        return ReasoningMCTSNode(prompt, question_prompt, self.gen_fn, self.reward_fn, self.depth + 1,
                                 self._r1_default, self._r_alpha, self._prompt_index, parent=self, r0=r0)

    def _create_children(self):
        self._visited = True
        self._calculate_reward()
        if self.is_terminal:
            return self.children
        questions, question_prompts, r0 = self.gen_fn(self.partial_solution, self.useful, self.depth)
        
        #? question is action, qp is next_state, r is reward
        for question, qp, r in zip(questions, question_prompts, r0):
            self.children.append(self._child_node(question, qp, r))
        return self.children

    def find_children(self):
        self.children = self.children or self._create_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        self.partial_solution, self._r1, self._ans_list = self.reward_fn(self.partial_solution, self.depth)

    def _static_terminal(self):
        return reach_terminal_subquestion(self.partial_solution, self._prompt_index)

    @property
    def is_terminal(self):
        return self._static_terminal() or self.reward < -1

    @property
    def reward(self):
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        return self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha)

    def print(self, mcts: MCTS, file=None):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        question = 'Q' + self.partial_solution.split(f'Question {self._prompt_index}')[-1].split('\n')[0]
        pprint(p1 + question)
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f} ; r0 : {self._r0:.3f}')
        if not self.visited:
            return
        answer = 'A' + self.partial_solution.split(f'Answer {self._prompt_index}')[-1].split('\n')[0]
        if self.reward < -1:
            if file is not None:
                pprint(prefix + question)
                pprint(prefix + answer)
            return
        if self.parent is not None:
            if file is not None:
                pprint(prefix + answer)
            match = re.match('.*The answer is (.*)\\.', answer)
            if match is not None:
                term = '\u25A1' if self.is_terminal else ''
                pprint(prefix + f'answer: {match[1]} ; ans_list: {self._ans_list} ; r1 : {self._r1:.3f}{term}')
        for child in self.children:
            child.print(mcts, file)
        if self.depth == 1:
            pprint("=" * 12)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state


def reasoning_mcts_search(question: str,
                          decompose_examples,
                          useful_examples,
                          world_model: QueryLM,
                          n_sample_subquestion,
                          temperature,
                          mcts_rollouts,
                          w_exp,
                          n_sample_confidence,
                          max_depth,
                          r_alpha,
                          r1_default,
                          eos_token_id,
                          speedup_confidence_batch_size=None):
    if speedup_confidence_batch_size is None:
        speedup_confidence_batch_size = n_sample_confidence

    overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', question)[1]
    overall_question = overall_question[0].upper() + overall_question[1:]   # capitalize the first letter
    question_group_id = decompose_examples['index']

    def gen_fn(partial_solution, useful, depth):
        #! self.partial_solution, self.useful, self.depth
        #? input: last action, current state, depth
        #? output: possible next actions, corresponding next states, corresponding rewards
        subquestion_prefix = decompose_examples["subquestion_prefix"].format(depth)
        eliciting_subquestions = partial_solution + subquestion_prefix
        overall_question_output = partial_solution + decompose_examples["overall_question_prefix"].format(depth, overall_question)

        if depth == max_depth:  
            candidate_partial_solutions = [overall_question_output]
        else:
            #! LLM generates a list of candidate sub-questions towards next depth (e.g. 4 possible Question 5.2, num_return_sequences=4, depth=2)
            candidate_partial_solutions = world_model.query_LM(eliciting_subquestions, do_sample=True, num_return_sequences=n_sample_subquestion,
                                                eos_token_id=eos_token_id, temperature=temperature)
            for i, candidate in enumerate(candidate_partial_solutions):
                if reach_terminal_subquestion(candidate, question_group_id):
                    candidate_partial_solutions[i] = overall_question_output

        # unique the output
        # set does not guarantee order ; dict guarantees insertion order
        candidate_partial_solutions = list(dict.fromkeys(candidate_partial_solutions))
        new_subquestions = [o.split(subquestion_prefix)[-1] for o in candidate_partial_solutions]     
        r0 = r0_fn(useful, new_subquestions, depth)

        useful_partial_trace = [
            useful + useful_examples["subquestion_prefix"].format(depth) + q 
            for q in new_subquestions]

        return candidate_partial_solutions, useful_partial_trace, r0

    def r0_fn(useful, new_subquestions, depth):
        """self-evaluation of helpfulness of a new subquestion"""
        inp = [useful + useful_examples["new_subquestion_prefix"].format(depth) +
               q.replace('Now we can answer the question: ', '') +
               useful_examples["answer_prefix"] for q in new_subquestions]
        yes_no = world_model.query_next_token(inp)
        return yes_no[:, 0] #! the probs of answer being "yes"

    def r1_fn(partial_solution, depth):
        """confidence"""
        if f'Question {question_group_id}.' not in partial_solution:
            return partial_solution, 0, []
        answer_prefix = decompose_examples["answer_prefix"].format(depth - 1)   #! Answer 5.x:
        eliciting_ans_to_subquestion = partial_solution + answer_prefix  

        direct_answer_dict = defaultdict(lambda: [])
        direct_answer_list = []
        num_sampling_attempts = 0
        while num_sampling_attempts < n_sample_confidence:
            model_output_list = world_model.query_LM(eliciting_ans_to_subquestion, do_sample=True,
                                                num_return_sequences=speedup_confidence_batch_size,
                                                eos_token_id=eos_token_id, temperature=temperature)
            num_sampling_attempts += speedup_confidence_batch_size
            for model_output in model_output_list:
                result = model_output.strip().split('\n')[-1]
                match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                if match is None:
                    continue
                direct_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                direct_answer_dict[direct_answer].append(model_output)
                direct_answer_list.append(direct_answer)
            if len(direct_answer_dict) == 0:
                continue
            sorted_direct_answer_dict = sorted(direct_answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_direct_answer_dict[0][1])
            if max_len < 2:
                continue
            if len(sorted_direct_answer_dict) < 2:
                break
            second_max_len = len(sorted_direct_answer_dict[1][1])
            if max_len >= len(direct_answer_dict) / 2 and max_len > second_max_len:
                break
        if len(direct_answer_dict) == 0:
            return output, -10, []
        selected_answer = sorted_direct_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
        r1 = max_len / len(direct_answer_list)
        return selected_answer, r1, direct_answer_list 

    def reward_fn(partial_solution, depth):
        return r1_fn(partial_solution, depth)

    decompose = decompose_examples["input"] + decompose_examples["question_prefix"] + question.strip() + "\n"
    useful = useful_examples["input"] + useful_examples["question_prefix"] + question.strip() + "\n"

    mcts = MCTS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max')
    root = ReasoningMCTSNode(decompose, useful, gen_fn, reward_fn,
                             depth=1, r1_default=r1_default, r_alpha=r_alpha, prompt_index=question_group_id)
    trajs, trees = [], []
    for _ in (pbar := trange(mcts_rollouts, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        mcts.rollout(root)
        # root.print(mcts)
        max_n, max_r = mcts.max_mean_terminal(root)
        trajs.append(traj := max_n.partial_solution.split('\n\n')[-1])
        output = re.findall('The answer is (.+).*\\.', traj)
        if len(output) == 0:
            temp_r = 'not found'
        else:
            temp_r = output[-1].replace(',', '')
        pbar.set_description(f'{max_r:.3f} {temp_r}')
        tree_copy = deepcopy(root)
        tree_copy.Q = dict(mcts.Q)
        tree_copy.N = dict(mcts.N)
        tree_copy.M = dict(mcts.M)
        trees.append(tree_copy)
        breakpoint()

    with io.StringIO() as f:
        root.print(mcts, file=f)
        tree = f.getvalue()
    return trajs, tree, trees
