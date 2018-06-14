""" This file contains functions for plotting attention map
"""
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import pandas as pd

def draw_attention(batch, ex_idx, attention_map, pred_ans_tokens):
    context_tokens = [x.strip() for x in " ".join(batch.context_tokens[ex_idx]).split(';')]
    print("length of context is {}".format(len(context_tokens)))
    attention_map = np.array(attention_map)[:len(context_tokens), : len(pred_ans_tokens)]
    for i, x in enumerate(context_tokens):
        print i, x
    sns.heatmap(attention_map)
    plt.xlabel(pred_ans_tokens)
    plt.ylabel(context_tokens)
    plt.show()

def plot_heatmap(predictions):
    df = pd.DataFrame(index=np.arange(300), columns=np.arange(20))
    df = df.fillna(-10)

    count_dict = {}
    for i in range(len(predictions)):
        pred = predictions[i].strip().split(' ')
        graph_len, route_len, em = [int(x) for x in pred[-3:]]
        key = (graph_len, route_len)
        if key not in count_dict:
            count_dict[key] = [em, 1]
        else:
            count_dict[key][0] += em
            count_dict[key][1] += 1
    for (graph_len, route_len) in count_dict:
        df.ix[graph_len, route_len] = 10 * float(count_dict[(graph_len, route_len)][0]) / count_dict[(graph_len, route_len)][1]

    sns.heatmap(df, cmap='coolwarm')
    plt.show()
    # plt.get_figure().savefig('../heatmap.png')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for preduction' )
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.prediction_file) as prediction_file:
        predictions = prediction_file.readlines()
    plot_heatmap(predictions)