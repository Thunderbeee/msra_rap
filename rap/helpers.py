from rap.models import QueryLlama
from dataclasses import dataclass
import matplotlib.pyplot as plt

def draw_hist(r0_path, r1_path):
    """
    r0_path is a list, each element in the list is a tuple. The first element of the tuple is r0_score, the second element of the tuple is whether it is predicted correctly (0 is Wrong, 1 is Correct).
    r1_path is a list, each element in the list is a tuple. The first element of the tuple is r1_score, the second element of the tuple is whether it is predicted correctly.
    the function `draw_hist` draws two histograms: the first histogram plot the distribution of r0_score and for each class (Wrong or Correct) has different color; the second histogram plot the distribution of r1_score and for each class (Wrong or Correct) has different color. 
    Save two plots figure in current directory
    Names as "distribution_r0.png" and "distribution_r1.png"
    """
    r0_scores, r0_correct = zip(*r0_path)
    r1_scores, r1_correct = zip(*r1_path)

    plt.hist([r0_scores[i] for i in range(len(r0_scores)) if r0_correct[i] == 0], bins=100, alpha=0.5, color='red', label='Wrong')
    plt.hist([r0_scores[i] for i in range(len(r0_scores)) if r0_correct[i] == 1], bins=100, alpha=0.5, color='green', label='Correct')

    plt.xlabel('r0_score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of r0_score')
    plt.savefig('distribution_r0.png')
    plt.show()

    plt.hist([r1_scores[i] for i in range(len(r1_scores)) if r1_correct[i] == 0], bins=100, alpha=0.5, color='red', label='Wrong')
    plt.hist([r1_scores[i] for i in range(len(r1_scores)) if r1_correct[i] == 1], bins=100, alpha=0.5, color='green', label='Correct')

    plt.xlabel('r1_score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of r1_score')
    plt.savefig('distribution_r1.png')
    plt.show()

@dataclass
class ExtraInfo:
    query_LM_counter: int = 0
    num_hit_max_depth: int = 0
    exec_time: float = 0.0
    max_depth_reached: int = 0
    path_r0: float = 0.0
    path_r1: float = 0.0
    path_reward: float = 0.0
    
    def reset(self):
        self.query_LM_counter = 0
        self.num_hit_max_depth = 0
        self.exec_time = 0.0
        self.max_depth_reached = 0
        self.path_r0 = 0.0
        self.path_r1 = 0.0
        self.path_reward = 0.0


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
