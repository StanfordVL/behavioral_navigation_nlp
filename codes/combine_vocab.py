'''
This script append extra words in vocab file B into vocab file A.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def combine(from_file, to_file):
    parent_dir = os.path.dirname(from_file)

    from_content = []
    to_content = []
    with open(from_file, 'r') as f:
        from_content = f.readlines()
    with open(to_file, 'r') as f:
        to_content = f.readlines()

    from_extra_set = set()
    to_set = set()

    for line in to_content:
        vocab = line.strip()
        to_set.add(vocab)
    for line in from_content:
        vocab = line.strip()
        if vocab not in to_set:
            from_extra_set.add(vocab)
    print("the extra vocabs that only appear in from vocab file is {}".format(from_extra_set))
    # with open(from_file + '.new', 'w') as f:
    # for vocab in from_extra_set:

    return


def main():
    if len(sys.argv) < 3:
        print("Usage: python count_length_distribution.py file")
        sys.exit(1)
    from_file = sys.argv[1]
    to_file = sys.argv[2]
    combine(from_file, to_file)


if __name__ == '__main__':
    main()