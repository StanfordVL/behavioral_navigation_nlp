from __future__ import print_function
from collections import Counter, defaultdict
import string
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Interface for summarizing all metrics
# input are two lists of strings of form [start] + [action list] + [goal]
def compute_all_metrics(pred_answer, true_answer):
    # because we only consider the accuracy of the actions, we remove first and last items.
    prediction_str = normalize_answer(" ".join(pred_answer[1:-1]))
    ground_truth_str = normalize_answer(" ".join(true_answer[1:-1]))
    em = exact_match_score(prediction_str, ground_truth_str)
    f1 = f1_score(prediction_str, ground_truth_str)
    ed = edit_distance(prediction_str.split(), ground_truth_str.split())
    if em > int(pred_answer[-1] == true_answer[-1]):
        print("weird thing happens, pred {}, true {}".format(pred_answer, true_answer))
    return f1, em, ed, pred_answer[-1] == true_answer[-1]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def edit_distance(s1, s2):
    """
    :param s1: list
    :param s2: list
    :return: edit distance of two lists
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def exact_match_score(prediction, ground_truth):
    return prediction == ground_truth

def rough_match_score(prediction, ground_truth):
    prediction = ' '.join(prediction.split(' '))
    ground_truth = ' '.join(ground_truth.split(' '))
    pred_list = prediction.split(' ')
    truth_list = ground_truth.split(' ')
    poss_correct = len(pred_list) == len(truth_list) or \
                   (len(pred_list) > len(truth_list) and pred_list[len(truth_list)] not in ['oor', 'ool'])
    return prediction[: len(ground_truth)] == ground_truth and poss_correct

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(ground_truth, predictions):
    f1_total = em_total = 0
    total = len(ground_truth)
    err_analysis = defaultdict(list)
    assert len(ground_truth) == len(predictions)

    for i in range(total):
        truth = ground_truth[i].strip().split(' ')[1:]
        pred = predictions[i].strip().split(' ')
        f1 = f1_score(predictions[i], " ".join(truth))
        em = exact_match_score(predictions[i], " ".join(truth))
        for j in range(len(truth)):
            err_analysis[j].append(j < len(pred) and truth[j] == pred[j])
        f1_total += f1
        em_total += em
    err_dist = np.zeros([len(err_analysis)])

    for k in err_analysis:
        err_dist[k] = sum(err_analysis[k]) / float(len(err_analysis[k]))
    plt.plot(err_dist)
    plt.xlabel("pos in the answer")
    plt.ylabel("accuracy")
    plt.show()
    exact_match = 100.0 * em_total / total
    f1 = 100.0 * f1_total / total
    print('exact_match: {}, f1: {}'.format(exact_match, f1))
    return

def evaluate_new(ground_truth, predictions):
    """
    :param ground_truth: a list of strings
    :param predictions: a list of strings
    :return: nil, side effect: print out the metrics value.
    """
    assert len(ground_truth) == len(predictions)
    f1_all = 0.0
    em_all = 0.0
    ed_all = 0.0
    gem_all = 0.0
    i = 0
    for (g, p) in zip(ground_truth, predictions):
        i += 1
        # print(i)
        pred_answer = p.strip().split(" ")
        true_answer = g.strip().split(" ")
        true_answer = [true_answer[0]] + true_answer[1::2] + [true_answer[-1]]
        f1, em, ed, gem = compute_all_metrics(pred_answer, true_answer)
        f1_all += f1
        em_all += em
        ed_all += ed
        gem_all += gem

    f1_all /= len(ground_truth)
    em_all /= len(ground_truth)
    ed_all /= len(ground_truth)
    gem_all /= len(ground_truth)
    print("f1 {}, em {}, ed {}, gem {}".format(f1_all, em_all, ed_all, gem_all))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for preduction' )
    parser.add_argument('true_file', help='True file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.true_file) as true_file:
        dataset = true_file.readlines()
    with open(args.prediction_file) as prediction_file:
        predictions = prediction_file.readlines()
    evaluate_new(dataset, predictions)