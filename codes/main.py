"""This file contains the entry point to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove, create_vocabulary, create_vocab_class

logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")
tf.app.flags.DEFINE_boolean("use_raw_graph", False, "Use vectors in unit of triplets to represent graph")
tf.app.flags.DEFINE_boolean("use_bidaf", False, "Use BiDAF attention, otherwise multiplicative attention")
tf.app.flags.DEFINE_boolean("show_start_tokens", False, "Whether to explicitly append start positions in graph.")
tf.app.flags.DEFINE_boolean("schedule_embed", False, "Whether to scheduled training in decoder's training stage.")
tf.app.flags.DEFINE_float("sampling_prob", 0.0, "the probability of sampling categorically from the output ids instead of reading directly from the inputs")
tf.app.flags.DEFINE_string("pred_method", "greedy", "Available sampling method in decoder's test time: greedy / sample / beam")

# sampling prob decay
tf.app.flags.DEFINE_float("sampling_prob_decay_factor", 0.99,
                          "Decay rate for exponential learning decay.")
tf.app.flags.DEFINE_float("sampling_prob_decay_steps", 100,
                          "If zero, no decaying is used. Otherwise, sampling prob decays this often "
                          "(sampling_prob = 1 - (1 - sampling_prob) * decay_factor ^ (global_step / decay_steps))")


# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.4, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 128, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 300, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 150, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("answer_len", 50, "The maximum answer length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")
tf.app.flags.DEFINE_integer("context_vocabulary_size", 200, "The maximum vocabulary size in context")
tf.app.flags.DEFINE_integer("ans_vocabulary_size", 200, "The maximum vocabulary size in answer")

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 100, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")
tf.app.flags.DEFINE_integer("print_num", 5, "how many examples to print out in show_examples mode")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("file_in_path", "dev", "For official_eval mode, evaluate which data choose from dev or test")
tf.app.flags.DEFINE_string("file_out_path", "predictions.txt", "Output path for official_eval mode. Defaults to predictions.txt")
tf.app.flags.DEFINE_boolean("write_out", False, "write out the prediction to file.")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Looking for model at %s..." % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_context_path = os.path.join(FLAGS.data_dir, "train.graph")
    train_qn_path = os.path.join(FLAGS.data_dir, "train.instruction")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.answer")
    dev_context_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".graph")
    dev_qn_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".instruction")
    dev_ans_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".answer")

    # Create vocabularies of the appropriate sizes for output answer.
    context_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.context" % FLAGS.context_vocabulary_size)
    # ans_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d." % FLAGS.ans_vocabulary_size)

    # initialize the vocabulary.
    context_vocab, rev_context_vocab = create_vocabulary(context_vocab_path, train_context_path,
                                                         FLAGS.context_vocabulary_size)
    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix, context_vocab, rev_context_vocab, context_vocab)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS(sys.argv), fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:
            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)

    elif FLAGS.mode == "show_examples":
        """ To show a few examples without attention map.
        """
        with tf.Session(config=config) as sess:
            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)
            # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            eval_context_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".graph")
            eval_qn_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".instruction")
            eval_ans_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".answer")
            _, _, _, _ = qa_model.check_f1_em(sess, eval_context_path, eval_qn_path, eval_ans_path, FLAGS.file_in_path,
                                           num_samples=FLAGS.print_num, print_to_screen=True)#, summary_writer=summary_writer)
            # summary_writer.close()

    elif FLAGS.mode == "demo":
        """ To show a few examples of attention map.
        """
        with tf.Session(config=config) as sess:
            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)
            # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            eval_context_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".graph")
            eval_qn_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".instruction")
            eval_ans_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".answer")
            qa_model.demo(sess, eval_context_path, eval_qn_path, eval_ans_path, FLAGS.file_in_path,
                                              num_samples=FLAGS.print_num,
                                              print_to_screen=True, shuffle=False)  # , summary_writer=summary_writer)

    elif FLAGS.mode == "official_eval":
        with tf.Session(config=config) as sess:
            if FLAGS.ckpt_load_dir:
                # Load model from ckpt_load_dir
                initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)
            else:
                # Load best model
                initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)
            eval_context_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".graph")
            eval_qn_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".instruction")
            eval_ans_path = os.path.join(FLAGS.data_dir, FLAGS.file_in_path + ".answer")
            f1, em, edit_dist, rem = qa_model.check_f1_em(sess, eval_context_path, eval_qn_path, eval_ans_path, FLAGS.file_in_path,
                                            num_samples=0, print_to_screen=False, write_out=FLAGS.write_out,
                                            file_out=FLAGS.file_out_path, shuffle=False)
            logging.info("F1 score: %f, EM score: %f, edit distance: %f, rough EM score: %f" % (f1, em, edit_dist, rem))
    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
