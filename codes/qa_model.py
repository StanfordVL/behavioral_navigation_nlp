"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import compute_all_metrics
from data_batcher import get_batch_generator
from pretty_print import print_example
from plot_utils import draw_attention
from modules import RNNEncoder, RNNDecoder, BasicAttn, masked_softmax, BiDAF
from vocab import PAD_ID, one_hot_converter, create_vocab_class
import utils

logging.basicConfig(level=logging.INFO)
SEP_TOKEN = ';'

class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix, ans2id, id2ans, context2id):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string) for instructions.
          word2id: dictionary mapping word (string) to word idx (int) for instructions.
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
          id2ans: dictionary mapping word idx (int) to word (string) for output routes.
          ans2id: dictionary mapping word (string) to word idx (int) for output routes.
          context2id: dictionary mapping word (string) to word idx (int) for the whole graph.
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.ans_vocab_size = len(ans2id)
        self.ans2id = ans2id
        self.id2ans = id2ans
        self.context2id = context2id

        # creating the vocab class, a class converting raw graph token ids into the id lists corresponding to triplets.
        # See Vocab class in vocab.py for more info. the following two variables are only used when not use_raw_graph
        self.graph_vocab_class = create_vocab_class(context2id)
        self.context_dimension_compressed = len(self.graph_vocab_class.all_tokens) + len(self.graph_vocab_class.nodes)

        # sampling_prob
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if FLAGS.sampling_prob_decay_steps == 0:
            self.sampling_prob = tf.Variable(float(FLAGS.sampling_prob), dtype=tf.float32, trainable=False)
        else:
            self.sampling_prob = 1 - tf.train.exponential_decay(1 - FLAGS.sampling_prob,
                                                            self.global_step,
                                                            FLAGS.sampling_prob_decay_steps,
                                                            FLAGS.sampling_prob_decay_factor,
                                                            staircase=True)
        # Add all parts of the graph
        with tf.variable_scope("QAModel",
                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix, ans2id, context2id)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len], name="context_ids")
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len], name="context_mask")
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len], name="qn_ids")
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len], name="qn_masks")
        self.ans_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.answer_len], name="ans_ids")
        self.ans_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.answer_len], name="ans_mask")
        self.context_embedding = tf.placeholder(tf.float32,
                                                shape=[None, self.FLAGS.context_len, self.context_dimension_compressed],
                                                name="context_embedding")
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

    def add_embedding_layer(self, emb_matrix, ans2id, context2id, use_same=True):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).dev
          ans2id: dict convert answer tokens to its token id
          context2id: dict convert context tokens to its token id.
          use_same: Default true. True if use same vocabulary for context and answer.
        """
        with vs.variable_scope("embeddings"):
            qn_embedding_matrix = tf.Variable(emb_matrix, dtype=tf.float32,
                                              name="qn_emb_matrix")  # shape (400002, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(qn_embedding_matrix, self.qn_ids)

            self.context_embedding_matrix = tf.Variable(tf.one_hot(range(len(context2id)), len(context2id)),
                                                      name="context_emb_matrix")
            if not use_same:
                self.ans_embedding_matrix = tf.Variable(tf.one_hot(range(len(ans2id)), len(ans2id)),
                                                       dtype=tf.float32, name="ans_emb_matrix")
            else:
                self.ans_embedding_matrix = tf.identity(self.context_embedding_matrix, name="ans_emb_matrix")

            if self.FLAGS.use_raw_graph:
                self.context_embs = embedding_ops.embedding_lookup(self.context_embedding_matrix, self.context_ids)
            else:
                self.context_embs = tf.identity(self.context_embedding)
            self.ans_embs = embedding_ops.embedding_lookup(self.ans_embedding_matrix, self.ans_ids)


    def build_graph(self):
        """Builds the main part of the graph for the model.
        """
        with vs.variable_scope("context"):
            context_encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = context_encoder.build_graph(self.context_embs,
                                                          self.context_mask)  # (batch_size, context_len, hidden_size*2)

        with vs.variable_scope("question"):
            question_encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            question_hiddens = question_encoder.build_graph(self.qn_embs,
                                                            self.qn_mask)  # (batch_size, question_len, hidden_size*2)
            question_last_hidden = tf.reshape(question_hiddens[:, -1, :], (-1, 2 * self.FLAGS.hidden_size))
            question_last_hidden = tf.contrib.layers.fully_connected(question_last_hidden,
                                                                     num_outputs=self.FLAGS.hidden_size)
        # Use context hidden states to attend to question hidden states

        # attn_output is shape (batch_size, context_len, hidden_size*2)
        # The following is BiDAF attention
        if self.FLAGS.use_bidaf:
            attn_layer = BiDAF(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS.hidden_size * 2)
            attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens,
                                                self.context_mask)  # (batch_size, context_len, hidden_size * 6)
        else: # otherwise, basic attention
            attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS.hidden_size * 2)
            _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens)
        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size*4)

        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size)

        decoder = RNNDecoder(self.FLAGS.batch_size, self.FLAGS.hidden_size, self.ans_vocab_size, self.FLAGS.answer_len,
                             self.ans_embedding_matrix, self.keep_prob, sampling_prob=self.sampling_prob,
                             schedule_embed=self.FLAGS.schedule_embed, pred_method=self.FLAGS.pred_method)
        (self.train_logits, self.train_translations, _), \
        (self.dev_logits, self.dev_translations, self.attention_results) = decoder.build_graph(blended_reps_final, question_last_hidden,
                                                                       self.ans_embs, self.ans_mask, self.ans_ids,
                                                                       self.context_mask)

    def add_loss(self):
        """
        Add loss computation to the graph.
        Defines:
          self.loss: scalar tensors of the loss in training time.
          self.dev_loss: scaler tensors of the loss in test time.
        """
        with vs.variable_scope("loss"):
            weights = tf.to_float(tf.not_equal(self.ans_ids, PAD_ID))  # [batch_size, context_len]

            # shift the weight right to include the end id
            batch_size = tf.shape(weights)[0]
            shift_val = tf.ones([batch_size, 1])

            self.new_ans_ids = tf.concat([self.ans_ids[:, 1:], tf.fill([batch_size, 1], 0)], 1)
            self.logits = self.train_logits
            weights = tf.concat([shift_val, weights], 1)[:, :-1]
            self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.new_ans_ids, weights=weights)
            tf.summary.scalar('train_loss', self.loss)
            tf.summary.scalar('sampling_prob', self.sampling_prob)

            if self.FLAGS.pred_method == 'beam':
                self.dev_logits = tf.Print(self.dev_logits[:, :, 0], [tf.shape(self.dev_logit), self.dev_logits[0, :, 0]])
                self.dev_loss = tf.cast(self.dev_logits[0, 0], tf.float32)
                return  
            dev_logits_len = tf.to_int32(tf.shape(self.dev_logits)[1])
            weights = tf.concat([weights[:, 1:], tf.fill([batch_size, 1], 0.0)], 1) 
            self.dev_loss = tf.contrib.seq2seq.sequence_loss(
                self.dev_logits, self.new_ans_ids[:, :dev_logits_len],
                weights=weights[:, :dev_logits_len])

    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_ids] = batch.ans_ids
        input_feed[self.ans_mask] = batch.ans_mask
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout  # apply dropout

        # if not use raw graph tokens
        if not self.FLAGS.use_raw_graph:
            input_feed[self.context_embedding] = batch.context_embeddings

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm, self.dev_loss]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm, dev_loss] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm, dev_loss

    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_ids] = batch.ans_ids
        input_feed[self.ans_mask] = batch.ans_mask
        if not self.FLAGS.use_raw_graph:
            input_feed[self.context_embedding] = batch.context_embeddings
        # Note: don't supply keep_prob here, so it will default to 1 i.e. no dropout
        output_feed = [self.dev_loss]
        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for output.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          traindist, probdist, alignment_history (for attention visualization), dev_logits
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_mask] = batch.ans_mask
        input_feed[self.ans_ids] = batch.ans_ids
        if not self.FLAGS.use_raw_graph:
            input_feed[self.context_embedding] = batch.context_embeddings
        output_feed = [self.train_translations, self.dev_translations, self.attention_results, self.dev_logits]
        [traindist, probdist, alignment_history, dev_logits] = session.run(output_feed, input_feed)
        return traindist, probdist, alignment_history, dev_logits

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        i = 0
        for batch in get_batch_generator(self.word2id, self.context2id, self.ans2id, dev_context_path,
                                         dev_qn_path, dev_ans_path, self.FLAGS.batch_size, self.graph_vocab_class,
                                         context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len,
                                         answer_len=self.FLAGS.answer_len, discard_long=False,
                                         use_raw_graph=self.FLAGS.use_raw_graph,
                                         show_start_tokens=self.FLAGS.show_start_tokens):
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)
            if i == 10:
                break
            i += 1

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc - tic)
        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss

    def demo(self, session, context_path, qn_path, ans_path, dataset, num_samples=10, print_to_screen=False,
                    write_out=False, file_out=None, shuffle=True):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (
            str(num_samples) if num_samples != 0 else "all", dataset))
        example_num = 0

        tic = time.time()
        ans_list = []
        graph_route_info = []

        for batch in get_batch_generator(self.word2id, self.context2id, self.ans2id, context_path,
                                         qn_path, ans_path, self.FLAGS.batch_size, self.graph_vocab_class,
                                         context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len,
                                         answer_len=self.FLAGS.answer_len, discard_long=False,
                                         use_raw_graph=self.FLAGS.use_raw_graph, shuffle=shuffle,
                                         show_start_tokens=self.FLAGS.show_start_tokens, output_goal=True):
            train_ids, pred_ids, dev_final_states, pred_logits = self.get_prob_dists(session, batch)
            start_ids = batch.ans_ids[:, 0].reshape(-1)

            if self.FLAGS.pred_method != 'beam':
                pred_ids, confidence_score, ans_str = output_route(start_ids, pred_logits, batch.context_tokens,
                                                                   self.ans2id, self.id2ans, self.FLAGS.answer_len)

            pred_ids = pred_ids.tolist()  # the output of using test network
            dev_attention_map = create_attention_images_summary(dev_final_states)
            print "dev_attention_map", dev_attention_map.shape
            dev_attention_map = dev_attention_map.eval().tolist()

            # the output of using training network, that the true input is fed as the input of the next RNN, for debug.
            for ex_idx, (pred_ans_list, true_ans_tokens, attention_map) in enumerate(
                    zip(pred_ids, list(batch.ans_tokens), dev_attention_map)):

                example_num += 1
                pred_ans_tokens = []
                for id in pred_ans_list:
                    if id == PAD_ID:
                        break
                    else:
                        pred_ans_tokens.append(self.id2ans[id])
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens[:])
                # Calculate metrics
                f1, em, edit_dist, rough_em = compute_all_metrics(pred_ans_tokens, true_ans_tokens)
                ans_list.append(pred_answer)

                if print_to_screen:
                    print_example(self.word2id, self.context2id, self.ans2id, batch.context_tokens[ex_idx],
                                  batch.qn_tokens[ex_idx], true_answer, pred_answer, f1, em, edit_dist,
                                  confidence_score[ex_idx])
                    # Draw attention map
                    draw_attention(batch, ex_idx, attention_map, pred_ans_tokens)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        toc = time.time()
        logging.info(
            "Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc - tic))
        if write_out:
            logging.info("Writing the prediction to {}".format(file_out))
            with open(file_out, 'w') as f:
                for line, extra_info in zip(ans_list, graph_route_info):
                    f.write(line + " " + " ".join(extra_info) + '\n')
            print("Wrote predictions to %s" % file_out)

        return

    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=10, print_to_screen=False,
                    write_out=False, file_out=None, shuffle=True):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (
            str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        ed_total = 0.
        rough_em_total = 0.
        example_num = 0

        tic = time.time()
        ans_list = []
        graph_route_info = []
        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, self.context2id, self.ans2id, context_path,
                                         qn_path, ans_path, self.FLAGS.batch_size, self.graph_vocab_class,
                                         context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len,
                                         answer_len=self.FLAGS.answer_len, discard_long=False,
                                         use_raw_graph=self.FLAGS.use_raw_graph, shuffle=shuffle,
                                         show_start_tokens=self.FLAGS.show_start_tokens, output_goal=True):
            train_ids, pred_ids, dev_final_states, pred_logits = self.get_prob_dists(session, batch)
            start_ids = batch.ans_ids[:, 0].reshape(-1)
            graph_length = np.sum(batch.context_mask, axis=1)

            if self.FLAGS.pred_method != 'beam':
                pred_ids, confidence_score, ans_str = verify_route(start_ids, pred_logits, batch.context_tokens,
                                                                   self.ans2id, self.id2ans, self.FLAGS.answer_len)

            pred_ids = pred_ids.tolist()  # the output of using test network
            for ex_idx, (pred_ans_list, true_ans_tokens) in enumerate(
                    zip(pred_ids, list(batch.ans_tokens))):
                example_num += 1
                pred_ans_tokens = []
                for id in pred_ans_list:
                    if id == PAD_ID:
                        break
                    else:
                        pred_ans_tokens.append(self.id2ans[id])
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens[:])

                # Calculate metrics
                f1, em, edit_dist, rough_em = compute_all_metrics(pred_ans_tokens, true_ans_tokens)

                f1_total += f1
                em_total += em
                ed_total += edit_dist
                rough_em_total += rough_em
                ans_list.append(pred_answer)
                graph_route_info.append((str(int(graph_length[ex_idx])), str(len(true_ans_tokens[1:-1])), str(int(em))))

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, self.context2id, self.ans2id, batch.context_tokens[ex_idx],
                                  batch.qn_tokens[ex_idx], true_answer, pred_answer, f1, em, edit_dist, confidence_score[ex_idx])

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break
        f1_total /= example_num
        em_total /= example_num
        ed_total /= example_num
        rough_em_total /= example_num

        toc = time.time()
        logging.info(
            "Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc - tic))
        if write_out:
            logging.info("Writing the prediction to {}".format(file_out))
            with open(file_out, 'w') as f:
                for line, extra_info in zip(ans_list, graph_route_info):
                    f.write(line + " " + " ".join(extra_info) + '\n')
            print("Wrote predictions to %s" % file_out)

        return f1_total, em_total, ed_total, rough_em_total

    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path,
              dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, self.context2id, self.ans2id, train_context_path,
                                             train_qn_path, train_ans_path, self.FLAGS.batch_size, self.graph_vocab_class,
                                             context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len,
                                             answer_len=self.FLAGS.answer_len, discard_long=False,
                                             use_raw_graph=self.FLAGS.use_raw_graph,
                                             show_start_tokens=self.FLAGS.show_start_tokens):
                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm, dev_loss = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss:  # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f, dev loss %.5f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time, dev_loss))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)

                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em, train_edit_dist, train_rem = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path,
                                                          "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f, Train Rough EM score: %f" % (
                        epoch, global_step, train_f1, train_em, train_rem))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)
                    write_summary(train_edit_dist, "train/edit_dist", summary_writer, global_step)

                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em, dev_edit_dist, dev_rem = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev",
                                                      num_samples=0)
                    logging.info(
                        "Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f, rough EM score: %f, edit distance: %f"
                        % (epoch, global_step, dev_f1, dev_em, dev_rem, dev_edit_dist))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)
                    write_summary(dev_edit_dist, "dev/edit-dist", summary_writer, global_step)
                    write_summary(dev_rem, "dev/Rough-EM", summary_writer, global_step)

                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        print("previous best em is {}, cur best em is {}.".format(best_dev_em, dev_em))
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)

            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))

        sys.stdout.flush()


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

def create_attention_images_summary(final_context_state):
  """create attention image and attention summary."""
  attention_images = (final_context_state)
  # Reshape to (batch, src_seq_len, tgt_seq_len,1)
  attention_images = tf.transpose(attention_images, [1, 2, 0])
  attention_images = tf.nn.l2_normalize(attention_images, axis=2, name='normalized_attention')
  # Scale to range [0, 255]
  attention_images *= 255
  return attention_images

def softmax(a):
  return np.exp(a) / np.sum(np.exp(a))


def verify_route(start_ids, pred_logits, context_tokens, ans2id, id2ans, ans_len):
    """
    Output a valid route.

    :param start_ids: [batch_size]
    :param pred_logits: [batch_size, length, vocab_size]
    :param context_tokens: [batch_size, context length]
    :param ans2id: dictionary of the ids of word tokens
    :param id2ans:
    :param ans_len: maximum answer length
    :return: a valid ans ids [batch_size, length]
    """
    batch_size = pred_logits.shape[0]
    mask = np.zeros(pred_logits.shape)
    ans = np.zeros((batch_size, ans_len), dtype=np.int16)
    scores = np.ones(batch_size, dtype=np.float16) # Compute the confidence.
    ans_with_nodes = [[] for _ in range(batch_size)] # records strings of action,node,action,...

    for i in range(batch_size):
        graph_str = " ".join(context_tokens[i])
        graph = utils.convert_map(graph_str)

        ansid_dict = {}
        for node in graph:
            ansid_dict[node] = [ans2id[edge] for edge in graph[node]]  # the ids of the valid actions
        cur_loc = id2ans[start_ids[i]]
        ans_with_nodes[i].append(cur_loc)
        for j in range(ans_len):
            if cur_loc not in ansid_dict:
                ans[i, j] = ans2id[cur_loc.split(" ")[0]]
                if j + 1 < len(ans[i]):
                    ans[i, j + 1] = PAD_ID
                break
            valid_ansids = ansid_dict[cur_loc]
            for ansid in valid_ansids:
                mask[i, j, ansid] = 1
            mask[i, j, PAD_ID] = 1
            exp_mask = (1 - mask[i, j, :]) * (-1e30)
            pred_logits[i, j, :] += exp_mask
            action_id= np.argmax(pred_logits[i, j, :])
            # Compute confidence score
            confidence_score = softmax(pred_logits[i, j, :])
            assert abs(np.sum(confidence_score) - 1) < 1e-5, "prob {}, action {}".\
                format(confidence_score, id2ans[action_id])
            scores[i] *= confidence_score[action_id]

            if action_id == PAD_ID:
                first_node = cur_loc.split(" ")[0]
                ans[i, j] = ans2id[first_node]
                if j + 1 < len(ans[i]):
                    ans[i, j+1] = PAD_ID
                break
            ans[i, j] = action_id
            action = id2ans[action_id]
            cur_loc = graph[cur_loc][action]

            # record the nodes and action strings.
            ans_with_nodes[i].append(action)
            ans_with_nodes[i].append(cur_loc)

    ans = np.concatenate((np.expand_dims(start_ids, axis=1), ans), axis=1)
    return ans, scores, ans_with_nodes

# TODO(Xiaoxue): Tidy the next function.
def output_route(start_ids, pred_logits, context_tokens, ans2id, id2ans, ans_len):
    """
    Output a valid route.

    :param start_ids: [batch_size]
    :param pred_logits: [batch_size, length, vocab_size]
    :param context_tokens: [batch_size, context length]
    :param ans2id: dictionary of the ids of word tokens
    :param id2ans:
    :param ans_len: maximum answer length
    :return: a valid ans ids [batch_size, length]
    """
    batch_size = pred_logits.shape[0]
    # ans_len = pred_logits.shape[1]
    ans = np.zeros((batch_size, ans_len), dtype=np.int16)
    scores = np.ones(batch_size, dtype=np.float16) # Compute the confidence.
    ans_with_nodes = [[] for _ in range(batch_size)] # records strings of action,node,action,...

    for i in range(batch_size):
        graph_str = " ".join(context_tokens[i])
        graph = utils.convert_map(graph_str)
        cur_loc = id2ans[start_ids[i]]
        ans_with_nodes[i].append(cur_loc)
        is_consistent = True
        for j in range(ans_len):
            # print(cur_loc)
            action_id= np.argmax(pred_logits[i, j, :])
            ################## Compute confidence score ################
            confidence_score = softmax(pred_logits[i, j, :])
            assert abs(np.sum(confidence_score) - 1) < 1e-5, "prob {}, action {}".format(confidence_score, id2ans[action_id])
            # print(confidence_score)
            scores[i] *= confidence_score[action_id]
            ############################################################

            if action_id == PAD_ID:
                first_node = cur_loc.split(" ")[0]
                ans[i, j] = ans2id[first_node]
                if j + 1 < len(ans[i]):
                    ans[i, j+1] = PAD_ID
                break
            ans[i, j] = action_id
            action = id2ans[action_id]
            if (action in graph[cur_loc]) and is_consistent:
                cur_loc = graph[cur_loc][action]
            else:
                is_consistent = False

            # record the nodes and action strings.
            ans_with_nodes[i].append(action)
            ans_with_nodes[i].append(cur_loc)

    ans = np.concatenate((np.expand_dims(start_ids, axis=1), ans), axis=1)
    return ans, scores, ans_with_nodes
