import json
import re


def judge_answer_multiarith(output, answer):
    output = re.findall('The answer is .*?([$ .0-9,\\-]+).*\\.', output)
    if len(output) == 0:
        output = ''
    else:
        output = output[-1].replace(',', '').replace('$', '').replace(' ', '')
    ret = output
    if '=' in output:
        output = output.split('=')[-1]
    try:
        output, answer = int(output), int(answer)
    except ValueError:
        try:
            output, answer = float(output), float(answer)
        except ValueError:
            pass
    return ret, output == answer


def get_multiarith_dataset(split):
    with open(f'data/multiarith/{split}.json') as f:
        examples = json.load(f)
    examples = [
        {"question": e["question"] + "\n", "answer": e["final_ans"]}
    for e in examples]
    print(f"{len(examples)} {split} examples")
    return examples