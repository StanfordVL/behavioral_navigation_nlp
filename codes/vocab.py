"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tensorflow.python.platform import gfile
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
import nltk


_PAD = b"<pad>"
_UNK = b"<unk>"
_SOS = b"<sos>"
_START_VOCAB = [_PAD, _UNK, _SOS]
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;()/-])")
_DIGIT_RE = re.compile(br"\d")
_ATTRIBUTE_RE = r'\d+[A-Za-z]+'
_NODE_RE = r'^[O|R|K|B|C|H|L]-\d+'
_ACT_RE = r'[A-Za-z]+'
_DISCARD_TOK = ['(', ')', 'nt', ';']


class Vocab:
    """
    Read in the class Vocab, it is a tidied class containing classified elements in graph
    self.node2id: dictionary converting the node token to its id
    self.nodes: list of all node tokens

    """
    def __init__(self, node2id, edge2id, flag2id):
        self.discard_tokens = _DISCARD_TOK
        self.node2id = node2id
        self.edge2id = edge2id
        self.flag2id = flag2id
        self.nodes = node2id.keys()
        self.edges = edge2id.keys()
        self.flags = flag2id.keys()
        self.all_tokens = self.flags + self.nodes + self.edges

    def tidy_in_triplet(self, tokens):
        """
        convert raw tokens into a id list of length [3*N]
        [first_node_id_list, edge_id_list, second_id_list, first_id_list, edge_id_list, second_id_list, ...]
        each entry in the list contains another list because node can be composed of many elements
        [[node_element0, node_element1, ...], [edge_element0, edge_element1, ...], [...], ...]
        :param tokens: a list of raw tokens in graph txt file
        :return: a list of lists
        """
        ids = []
        for (i, w) in enumerate(tokens):
            if w in self.nodes:
                if (i == 0) or (tokens[i - 1] in [';', ')']) or (tokens[i - 1] in self.edges):
                    ids.append([self.node2id[w]])
                else:
                    ids[-1].append(self.node2id[w])
            elif w in self.edges:
                if (tokens[i - 1] in self.flags) or (tokens[i - 1] in self.nodes):
                    ids.append([self.edge2id[w]])
                else:
                    ids[-1].append(self.edge2id[w])
            elif w in ['l', 'r']:
                w = "-".join(tokens[i - 1: i + 1])
                if w in self.edges:
                    ids[-1].append(self.edge2id[w])
                else:
                    ids[-1].append(UNK_ID)
            elif (w in self.discard_tokens) or (re.match(_ATTRIBUTE_RE, w)):
                if w == ';':
                    assert len(ids) % 3 == 0, "error in tidy_in_triplet, can't be divided by 3"
                continue
            elif w not in self.all_tokens:
                raise ValueError("new token %s in graph representation."%w)
        return ids

def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5)  # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception(
                    "You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (
                    glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word


def one_hot_converter(vec_len):
    one_hot_embed = np.zeros((vec_len, vec_len))
    np.fill_diagonal(one_hot_embed, 1)
    return one_hot_embed

def instruction_tokenizer(sentence):
    """
    A special tokenizer for instructions.
    Turn into lower case and split Office-1 or office1 into "Office 1",
    :param sentence: instructions (natural language)
    :return: a list of tokens
    """
    words = []
    prepocessed_sen_list = preprocess_instruction(sentence.strip())
    for space_separated_fragment in prepocessed_sen_list:
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]

def preprocess_instruction(sentence):
    # change "office-12" or "office12" to "office 12"
    # change "12-office" or "12office" to "12 office"
    _WORD_NO_SPACE_NUM_RE = r'([A-Za-z]+)\-?(\d+)'
    _NUM_NO_SPACE_WORD_RE = r'(\d+)\-?([A-Za-z]+)'
    new_str = re.sub(_WORD_NO_SPACE_NUM_RE, lambda m: m.group(1) + ' ' + m.group(2), sentence)
    new_str = re.sub(_NUM_NO_SPACE_WORD_RE, lambda m: m.group(1) + ' ' + m.group(2), new_str)
    lemma = nltk.wordnet.WordNetLemmatizer()
    # correct common typos.
    correct_error_dic = {'rom': 'room', 'gout': 'go out', 'roo': 'room',
                         'immeidately': 'immediately', 'halway': 'hallway',
                         'office-o': 'office 0', 'hall-o': 'hall 0', 'pas': 'pass',
                         'offic': 'office', 'leftt': 'left', 'iffice': 'office'}
    for err_w in correct_error_dic:
        find_w = ' ' + err_w + ' '
        replace_w = ' ' + correct_error_dic[err_w] + ' '
        new_str = new_str.replace(find_w, replace_w)
    sen_list = []
    # Lemmatize words
    for word in new_str.split(' '):
        try:
            word = lemma.lemmatize(word)
            if len(word) > 0 and word[-1] == '-':
                word = word[:-1]
            if word:
                sen_list.append(word)
        except UnicodeDecodeError:
            continue
            # print("unicode error ", word, new_str)
    return sen_list

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def create_vocab_class(vocab_dict):
    """
    Convert the raw tokens in graph to
    To organize the vocab into two groups: node group, action + attribute group
    :param vocab_dict:
    :param rev_vocab:
    :return: A vocab class holding all the information necessary for training
    """
    rev_vocab = vocab_dict.keys()
    new_vocab_dic = {"node": ["S", "N", "E", "W"], "edge": [], "flag": _START_VOCAB}
    for vocab in rev_vocab:
        if vocab in 'lrSNEW':
            continue
        elif re.match(_NODE_RE, vocab):
            new_vocab_dic["node"].append(vocab)
        elif re.match(_ATTRIBUTE_RE, vocab):
            new_vocab_dic["edge"].append(vocab + '-l')
            new_vocab_dic["edge"].append(vocab + '-r')
        elif re.match(_ACT_RE, vocab) and vocab != 'nt':
            new_vocab_dic["edge"].append(vocab)
    node2id = dict([(x, y) for (y, x) in enumerate(new_vocab_dic['node'])])
    edge2id = dict([(x, y) for (y, x) in enumerate(new_vocab_dic['edge'])])
    flag2id = dict([(x, y) for (y, x) in enumerate(new_vocab_dic['flag'])])

    return Vocab(node2id, edge2id, flag2id)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")
        rev_vocab = vocab_list # a list contain all the tokens
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])  # key is the token, value is index
        return vocab, rev_vocab
    else:
        print "Skipping generating vocabulary file for {}".format(vocabulary_path)
        return initialize_vocabulary(vocabulary_path)

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)