import os
import json


def avg(list):
    return sum(list) / len(list)


def analyze(src_path):
    query_LM_counter_list, num_hit_max_depth_list, exec_time_list = [], [], []
    num_correct, num_examples = 0, 0
    for file in os.listdir(src_path):
        if "json" not in file:
            continue
        
        num_examples += 1
        
        file_path = os.path.join(src_path, file)
        with open(file_path, "r") as js:
            list_of_rollouts = json.load(js)
        assert len(list_of_rollouts) == 10
        last_rollout = list_of_rollouts[-1]
        query_LM_counter_list.append(last_rollout["query_LM_counter"])
        num_hit_max_depth_list.append(last_rollout["num_hit_max_depth"])
        exec_time_list.append(last_rollout["exec_time"])
        if last_rollout["correct"] == True:
            num_correct += 1
    
    print(f"""
        ==> avg. query LM: {avg(query_LM_counter_list)} \n
        ==> avg. hit max depth: {avg(num_hit_max_depth_list)} \n
        ==> avg. execution time: {avg(exec_time_list)} \n
        ==> accuracy: {num_correct / num_examples}\n\n
    """)
    

if __name__ == "__main__":
    src_path = [
                "/home/xyyue/zangwei/mingyuan/rap/logs/gsm8k_mcts_llama-2-7b/2023-1102-0103",
                "/home/xyyue/zangwei/mingyuan/rap/logs/gsm8k_mcts_llama-2-7b/2023-1102-1919", #multiarith
                "/home/xyyue/zangwei/mingyuan/rap/logs/gsm8k_mcts_llama-2-7b-chat/2023-1102-0545",
                "/home/xyyue/zangwei/mingyuan/rap/logs/gsm8k_mcts_llama-2-7b-chat/2023-1102-1744", #multiarith
                ]
    for src in src_path:
        analyze(src)