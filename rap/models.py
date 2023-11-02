from abc import ABC, abstractmethod

import torch


class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt):
        pass


class QueryLlama(QueryLM):
    def __init__(self, model, tokenizer, max_response_length, log_file) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = 1
        self.yes_no = self.tokenizer.encode('Yes No')
        self.easy_medium_hard = self.tokenizer.encode('easy medium hard')
        self._1_to_7 = self.tokenizer.encode('one two three four five six seven')
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
            encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            results = self.model.generate(encoded_prompt, max_length=self.max_response_length+len(encoded_prompt[0]), temperature=temperature, num_return_sequences=end - start)
            
            if len(results.shape) > 2:
                results.squeeze_()
            for generated_sequence in results:
                generated_sequence = generated_sequence.tolist()
                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                all_results.append(text)
            # text = self.tokenizer.decode(results, clean_up_tokenization_spaces=True)
            self.query_LM_counter += 1
            
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
            import pdb; pdb.set_trace()
            tokens = self.tokenizer.encode(prompt)
            tokens = torch.tensor([tokens]).cuda().long()
            output = self.model.forward(tokens).logits[:, -1, :]
            self.query_LM_counter += 1
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

    @torch.no_grad()
    def query_difficulty_word(self, model_input: str):
        tokens = self.tokenizer.encode(model_input)
        tokens = torch.tensor([tokens]).cuda().long()
        import pdb; pdb.set_trace()
        output, h = self.model.forward(tokens)  #! output: [1, 32000]
        self.query_LM_counter += 1
        filtered = output[:, self.easy_medium_hard]
        dist = torch.softmax(filtered, dim=-1)  #! dist: [1, 3]
        max_id = torch.argmax(dist, dim=-1, keepdim=False) #! [1]
        return ["easy", "medium", "hard"][max_id.item()]
    
    @torch.no_grad()
    def query_difficulty_number(self, model_input: str):
        tokens = self.tokenizer.encode(model_input)
        tokens = torch.tensor([tokens]).cuda().long()
        output, h = self.model.forward(tokens, start_pos=0)  #! output: [1, 32000]
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
