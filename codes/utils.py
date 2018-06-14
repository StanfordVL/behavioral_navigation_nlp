"""This file contains functions for utility."""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import collections
import argparse

logging.basicConfig(level=logging.INFO)

EDGES = ['oor', 'ool', 'oio', 'lt', 'rt', 'sp', 'chs', 'chr', 'chl', 'cf', 'iol', 'ior']


def split_triplet(triplet_str):
    elements = triplet_str.split(' ')
    first_node = []
    edge = []
    second_node = []
    attribute = ''
    if '(' in elements:
        start, end = elements.index('('), elements.index(')')
        attribute = " ".join(elements[start: end + 1])
        elements = elements[:start] + elements[end + 1:]
    for ele in elements:
        if ele in EDGES:
            edge.append(ele)
        elif not edge:
            first_node.append(ele)
        else:
            second_node.append(ele)
    assert len(edge) == 1, "edge should only contain one element {}".format(elements)
    return " ".join(first_node), " ".join(edge), " ".join(second_node)


def convert_map(graph):
    tidied_map = collections.defaultdict(dict)
    for triplet in graph.strip().split(';'):
        triplet = triplet.strip()
        if not triplet:
            continue
        start, edge, end = split_triplet(triplet)
        start, edge, end = start.strip(), edge.strip(), end.strip()
        tidied_map[start][edge] = end
    return tidied_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test convert_map')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.prediction_file) as prediction_file:
        predictions = prediction_file.readlines()
    map_dict = convert_map(predictions)
    print(map_dict)
