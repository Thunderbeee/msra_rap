from abc import ABC, abstractmethod

import torch

from llama import LLaMA


class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt):
        pass


# class QueryHfModel(QueryLM):
#     # This is not well-tested. Please use LLaMA if possible.
#     def query_next_token(self, prompt):
#         raise NotImplementedError

#     def __init__(self, model, tokenizer, max_response_length, device):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = device
#         self.n_examples = 1
#         self.max_response_length = max_response_length

#     def query_LM(self, prompt, **gen_kwargs):
#         with torch.no_grad():
#             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#             # print("input length", len(inputs))
#             # Generate
#             generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=self.max_response_length, **gen_kwargs)
#             text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         return text


class QueryLlama(QueryLM):
    def __init__(self, llamamodel, max_response_length, log_file) -> None:
        self.llamamodel = llamamodel
        self.tokenizer = self.llamamodel.tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = llamamodel.model.params.max_batch_size
        self.yes_no = self.tokenizer.encode('Yes No', bos=False, eos=False)
        self.easy_medium_hard = self.tokenizer.encode('easy medium hard', bos=False, eos=False)
        self._1_to_7 = self.tokenizer.encode('one two three four five six seven', bos=False, eos=False)
        self.query_LM_counter = 0
    
    def reset_counter(self):
        self.query_LM_counter = 0

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        """
        input: prompt
        output: a list of outputs, length = num_return_sequences
        """
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            results = self.llamamodel.generate([prompt] * (end - start), max_gen_len=self.max_response_length, temperature=temperature, eos_token_id=eos_token_id)
            self.query_LM_counter += 1
            all_results.extend(results)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results

    @torch.no_grad()
    def query_next_token(self, prompts):
        #! prompts: a list of strings ending with "Is the new question useful?"
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda().long()
            output, h = self.llamamodel.model.forward(tokens, start_pos=0)
            self.query_LM_counter += 1
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

    @torch.no_grad()
    def query_difficulty_word(self, model_input: str):
        tokens = self.tokenizer.encode(model_input, bos=True, eos=False)
        tokens = torch.tensor([tokens]).cuda().long()
        output, h = self.llamamodel.model.forward(tokens, start_pos=0)  #! output: [1, 32000]
        self.query_LM_counter += 1
        filtered = output[:, self.easy_medium_hard]
        dist = torch.softmax(filtered, dim=-1)  #! dist: [1, 3]
        max_id = torch.argmax(dist, dim=-1, keepdim=False) #! [1]
        return ["easy", "medium", "hard"][max_id.item()]
    
    @torch.no_grad()
    def query_difficulty_number(self, model_input: str):
        tokens = self.tokenizer.encode(model_input, bos=True, eos=False)
        tokens = torch.tensor([tokens]).cuda().long()
        output, h = self.llamamodel.model.forward(tokens, start_pos=0)  #! output: [1, 32000]
        self.query_LM_counter += 1
        filtered = output[:, self._1_to_7]
        dist = torch.softmax(filtered, dim=-1)  #! dist: [1, 5]
        print("Prob dist is: ", dist)
        assert dist.shape[1] == 7, dist
        max_id = torch.argmax(dist, dim=-1, keepdim=False) #! [1]
        try:
            ans = [1, 2, 3, 4, 5, 6, 7][max_id.item()]
            return ans
        except:
            print(max_id.item())
            breakpoint()
        # return [2, 3, 4, 5, 6][max_id.item()]
