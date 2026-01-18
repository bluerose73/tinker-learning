import re
import argparse
import json

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True, help="Path to the model predictions file.")
    parser.add_argument("--gt-file", type=str, required=True, help="Path to the ground truth file.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load ground truth data
    gt_data = {}
    with open(args.gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            gt_data[example["question"]] = example
    
    # Load predictions and evaluate
    correct = 0
    format_correct = 0
    total = 0
    
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            pred_example = json.loads(line)
            question = pred_example["question"]
            
            if question not in gt_data:
                print(f"Warning: Question not found in ground truth: {question[:50]}...")
                continue
            
            gt_example = gt_data[question]
            
            # Check if the answer can be parsed
            pred_answer = extract_answer(pred_example["answer"])
            if pred_answer != INVALID_ANS:
                format_correct += 1
            
            # Check if the answer is correct
            if is_correct(pred_example["answer"], gt_example):
                correct += 1
            
            total += 1
    
    # Calculate and print metrics
    accuracy = correct / total * 100 if total > 0 else 0
    format_rate = format_correct / total * 100 if total > 0 else 0
    print(f"Total examples: {total}")
    print(f"Format correct: {format_correct}")
    print(f"Format correct rate: {format_rate:.2f}%")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()