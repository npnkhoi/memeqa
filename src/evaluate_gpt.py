import json
import argparse

def evaluate(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    
    # Calculate accuracy of each type and overall accuracy
    overall_correct = 0
    overall_empty = 0
    overall_total = 0
    type_correct = {}
    type_empty = {}
    type_total = {}

    for qid, result in data['per_example'].items():
        qtype = qid.split('/')[0]
        answer = result['answer']
        trimmed_output = result['trimmed_output']

        # Check if the answer is correct
        if trimmed_output == answer:
            overall_correct += 1
            type_correct[qtype] = type_correct.get(qtype, 0) + 1
        elif trimmed_output == "":
            overall_empty += 1
            type_empty[qtype] = type_empty.get(qtype, 0) + 1
        overall_total += 1
        type_total[qtype] = type_total.get(qtype, 0) + 1
    
    order = [
        "viz_id", "bk_id", "tar_id", "deriv_id", "tar_sent_id", 
        "tar_act_id", "deriv_comp", "intent_comp", "intent_id", 
        "cond_act_id", "cond_sent_id"
    ]
    ordered_accuracies = {qtype: 0.0 for qtype in order}

    for qtype in order:
        if qtype in type_correct:
            total = type_total[qtype]
            empty = type_empty.get(qtype, 0)
            accuracy = type_correct[qtype] / (total - empty) * 100 if total > 0 else 0
            ordered_accuracies[qtype] = float(accuracy)

    for qtype in order:
        print(f"{qtype}: {ordered_accuracies[qtype]:.3}")
    print(f"Macro Average Accuracy: {sum(ordered_accuracies.values()) / len(ordered_accuracies.values()):.3}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GPT-4O predictions.')
    parser.add_argument('fn', type=str, help='Path to the JSON file with predictions.')
    args = parser.parse_args()
    evaluate(args.fn)
    