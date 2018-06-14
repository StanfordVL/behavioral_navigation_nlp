"""This file contains functions to pretty-print a SQuAD example"""

from colorama import Fore, Back, Style
from vocab import _PAD

# See here for more colorama formatting options:
# https://pypi.python.org/pypi/colorama


def yellowtext(s):
    """Yellow text"""
    return Fore.YELLOW + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def bluetext(s):
    """Blue text"""
    return Fore.BLUE + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def magentatext(s):
    """Yellow text"""
    return Fore.MAGENTA + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def greentext(s):
    """Green text"""
    return Fore.GREEN + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redtext(s):
    """Red text"""
    return Fore.RED + Style.BRIGHT + s + Style.RESET_ALL + Fore.RESET

def redback(s):
    """Red background"""
    return Back.RED + s + Back.RESET

def magentaback(s):
    """Magenta background"""
    return Back.MAGENTA + s + Back.RESET



def print_example(word2id, context2id, ans2id, context_tokens, qn_tokens,
                  true_answer, pred_answer, f1, em, edit_dist, confidence_prob):
    """
    Pretty-print the results for one example.

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_tokens, qn_tokens: lists of strings, no padding.
        Note these do *not* contain UNKs.
      true_ans_start, true_ans_end, pred_ans_start, pred_ans_end: ints
      true_answer, pred_answer: strings
      f1: float
      em: bool
    """
    # Get the length (no padding) of this context
    curr_context_len = len(context_tokens)

    # Highlight out-of-vocabulary tokens in context_tokens
    context_tokens = [w if w in context2id else "_%s_" % w for w in context_tokens]

    # Highlight out-of-vocabulary tokens in qn_tokens
    qn_tokens = [w if w in word2id else "_%s_" % w for w in qn_tokens]

    # Highlight the true answer green.
    # If the true answer span isn't in the range of the context_tokens, then this context has been truncated
    truncated = False

    # Print out the context
    print "%s is true answer, %s is wrong prediction, _underscores_ are unknown tokens. Length: %i triplets" \
          % (greentext("green text"), redtext("green text"), context_tokens.count(";"))

    # print "CONTEXT: (%s is true answer, %s is predicted start, %s is predicted end, _underscores_ are unknown tokens). Length: %i" % (greentext("green text"), magentaback("magenta background"), redback("red background"), len(context_tokens))
    print " ".join(context_tokens)

    # Print out the question, true and predicted answer, F1 and EM score
    question = " ".join(qn_tokens)

    print magentatext("{:>20}: {}".format("QUESTION", question))
    if truncated:
        print greentext("{:>20}: {}".format("TRUE ANSWER", true_answer))
        print redtext("{:>22}(True answer was truncated from context)".format(""))
    else:
        print greentext("{:>20}: {}".format("TRUE ANSWER", true_answer))
    if f1 == 1:
        print greentext("{:>20}: {}".format("PREDICTED ANSWER", pred_answer))
    else:
        print redtext("{:>20}: {}".format("PREDICTED ANSWER", pred_answer))
    print yellowtext("{:>20}: {:4.3f}".format("F1 SCORE ANSWER", f1))
    print yellowtext("{:>20}: {}".format("EM SCORE", em))
    print yellowtext("{:>20}: {}".format("EDIT DISTANCE SCORE", edit_dist))
    print yellowtext("{:>20}: {}".format("CONFIDENE SCORE", confidence_prob))
    print ""
