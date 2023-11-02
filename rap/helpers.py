from rap.models import QueryLlama
from dataclasses import dataclass


@dataclass
class ExtraInfo:
    query_LM_counter: int = 0
    num_hit_max_depth: int = 0
    exec_time: float = 0.0
    max_depth_reached: int = 0
    
    def reset(self):
        self.query_LM_counter = 0
        self.num_hit_max_depth = 0
        self.exec_time = 0.0
        self.max_depth_reached = 0


difficulty_prefix = "The level of difficulty is:"
num_subq_prefix = "The number of subquestions we need is:"

eval_difficulty_prompt = f"""
Given a question, analyze the difficulty of the question, and output 'easy', 'medium', or 'hard', and a step-by-step reason.
Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody? {difficulty_prefix} medium\n
Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained? {difficulty_prefix} easy\n
Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner. {difficulty_prefix} hard\n
Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn? {difficulty_prefix} easy\n
"""

eval_num_subq_prompt = f"""
Given a question, analyze the difficulty of the question, and output 'easy', 'medium', or 'hard', and a step-by-step reason.
Question 1: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice as 30 years old, how old is Kody? {num_subq_prefix} five\n
Question 2: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies before two of the fireflies flew away. How many fireflies remained? {num_subq_prefix} two\n
Question 3: Ali has four $10 bills and six $20 bills that he saved after working for Mr. James on his farm. Ali gives her sister half of the total money he has and uses 3/5 of the remaining amount of money to buy dinner. Calculate the amount of money he has after buying the dinner. {num_subq_prefix} six\n
Question 4: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn? {num_subq_prefix} three\n
"""


def determine_max_depth(model: QueryLlama, question):
    # model_input = eval_difficulty_prompt + "Question 5: " + question.strip() + f" {difficulty_prefix} "
    # difficulty = model.query_difficulty_word(model_input)
    # difficulty2depth = {"easy": 3, "medium": 5, "hard": 6}
    # max_depth = difficulty2depth[difficulty]
    
    model_input = eval_num_subq_prompt + "Question 5: " + question.strip() + f" {num_subq_prefix} "
    max_depth = model.query_difficulty_number(model_input)
    return max_depth
