"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

from vocab import SOS_ID, PAD_ID

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1)  # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs,
                                                                  input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class RNNDecoder(object):
    """ Decode a sequence using a RNN
    """
    def __init__(self, batch_size, hidden_size, tgt_vocab_size, max_decoder_length, embeddings, keep_prob, sampling_prob,
                 schedule_embed=False, pred_method='greedy'):
        self.hidden_size = hidden_size
        self.projection_layer = tf.layers.Dense(
            tgt_vocab_size, use_bias=False)
        self.rnn_cell = rnn_cell.GRUCell(hidden_size)
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.start_id = SOS_ID
        self.end_id = PAD_ID
        self.tgt_vocab_size = tgt_vocab_size
        self.max_decoder_length = max_decoder_length
        self.keep_prob = keep_prob
        self.schedule_embed = schedule_embed
        self.pred_method = pred_method
        self.beam_width = 9
        self.sampling_prob = sampling_prob

    def build_graph(self, attention_embeds, encoder_state, decoder_emb_inputs, ans_masks, ans_ids, context_masks):
        """
        Inputs:
          attention_embeds: Tensor shape (batch_size, seq_len, hidden_size)
          encoder_state: Tensor shape (batch_size, hidden_size)
            NOTE(Xiaoxue): bug here, need to add another RNN to convert the states from bidirectional encoder.
          decoder_emb_inputs: Tensor shape (batch_size, output_seq_len, tgt_vocab_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.contrib.seq2seq.dynamic_decode doesn't iterate through masked steps.

        Returns:
          train_outputs: tuple of (Tensor shape (batch_size, seq_len, hidden_size), Tensor shape (batch_size, seq_len))
            for Training graph.

          pred_outputs: tuple of (Tensor shape (batch_size, seq_len, hidden_size), Tensor shape (batch_size, seq_len))
            for Testing graph.
        """
        with vs.variable_scope("RNNDecoder"):
            self.batch_size = tf.shape(decoder_emb_inputs)[0]

            # We give the first token, the start location, though usually in seqtoseq model, we append a start token.
            # start_ids = tf.fill([self.batch_size], self.start_id, name="start_tokens")
            start_ids = tf.cast(tf.reshape(ans_ids[:, 0], [-1]), tf.int32)
            # train_output = tf.concat([tf.expand_dims(tf.one_hot(start_ids, self.tgt_vocab_size), 1), decoder_emb_inputs], 1)
            train_output = decoder_emb_inputs
            # decoder_lengths = tf.reduce_sum(ans_masks, reduction_indices=1)
            memory_lengths = tf.reduce_sum(context_masks, reduction_indices=1)
            '''
            decoder_lengths = tf.Print(decoder_lengths,
                                       [tf.shape(decoder_lengths), tf.shape(train_output), train_output[0, :]],
                                       "decoder length and train_output shapes are ", summarize=200)
            '''
            decoder_lengths = tf.fill([self.batch_size], self.max_decoder_length)
            if not self.schedule_embed:
                train_helper = tf.contrib.seq2seq.TrainingHelper(train_output, decoder_lengths, time_major=False)
            else:
                # sampling_prob = tf.Variable(self.sampling_prob, dtype=tf.float32)
                train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(train_output, decoder_lengths, self.embeddings, self.sampling_prob, time_major=False)
            pred_start_ids = tf.reshape(ans_ids[:, 0], [-1])
            # pred_start_ids = tf.fill([self.batch_size], self.start_id)
            if self.pred_method == 'greedy':
                pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embeddings, start_tokens=tf.cast(pred_start_ids, tf.int32), end_token=self.end_id)
            elif self.pred_method == 'sample':
                pred_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    self.embeddings, start_tokens=tf.cast(pred_start_ids, tf.int32), end_token=self.end_id)
            elif self.pred_method == 'beam':
                tiled_attention_embeds = tf.contrib.seq2seq.tile_batch(
                    attention_embeds, multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    encoder_state, multiplier=self.beam_width)
                tiled_memory_length = tf.contrib.seq2seq.tile_batch(
                    memory_lengths, multiplier=self.beam_width)
                pred_helper = None
            else:
                raise ValueError('the prediction method is not support.')
            def decode(helper, scope, reuse=None, is_infer=False):
                with vs.variable_scope(scope, reuse=reuse):
                    # cell_input_fn = lambda inputs, attention: attention
                    cell_input_fn = lambda inputs, attention: tf.nn.dropout(tf.concat([inputs, attention], -1),
                                                                            self.keep_prob)
                    if is_infer == True and self.pred_method == 'beam':
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            num_units=self.hidden_size,
                            memory=tiled_attention_embeds,
                            memory_sequence_length=tiled_memory_length)
                        tiled_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                            self.rnn_cell, attention_mechanism, cell_input_fn=cell_input_fn,
                            attention_layer_size=self.hidden_size, alignment_history=False)
                        tiled_initial_state = tiled_attention_cell.zero_state(batch_size=self.batch_size * self.beam_width ,
                                                                  dtype=tf.float32).clone(cell_state=tiled_encoder_final_state)
                        start_ids = tf.cast(pred_start_ids, tf.int32)
                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=tiled_attention_cell,
                            embedding=self.embeddings,
                            start_tokens=start_ids,
                            end_token=self.end_id,
                            initial_state=tiled_initial_state,
                            beam_width=self.beam_width,
                            output_layer=self.projection_layer,
                            length_penalty_weight=0.0)
                        final_outputs, _final_state, _final_sequence_lengths = \
                            tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False,
                                                              maximum_iterations=self.max_decoder_length)
                        # if beam search, can not output the logits and calculate loss
                        logits = final_outputs.beam_search_decoder_output.scores
                        print('beam search decoder', logits.shape)
                        final_outputs_ids = final_outputs.predicted_ids[:, :, 0]
                    else:
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            self.hidden_size, attention_embeds,
                            memory_sequence_length=memory_lengths)

                        if is_infer:
                            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                self.rnn_cell, attention_mechanism, cell_input_fn=cell_input_fn,
                                attention_layer_size=self.hidden_size, alignment_history=True)
                        else:
                            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                self.rnn_cell, attention_mechanism, cell_input_fn=cell_input_fn,
                                attention_layer_size=self.hidden_size, alignment_history=False)
                        initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)

                        # Decoder and decode
                        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state,
                                                                  output_layer=self.projection_layer)
                        final_outputs, _final_state, _final_sequence_lengths = \
                            tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                          maximum_iterations=self.max_decoder_length)
                        final_outputs_ids = final_outputs.sample_id
                        logits = final_outputs.rnn_output

                    if is_infer:
                        alignment_history = _final_state.alignment_history.stack()
                    else:
                        alignment_history = tf.constant(1, dtype=tf.int32)
                    return logits, final_outputs_ids, alignment_history
            # This function returns two kinds of graphs (training graph and test graph)
            train_outputs = decode(train_helper, 'decode')
            pred_outputs = decode(pred_helper, 'decode', reuse=True, is_infer=True)

        return train_outputs, pred_outputs


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):
            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1,
                                                       activation_fn=None)  # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2])  # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):
            # Calculate attention distribution
            W = tf.get_variable("W", shape=[self.key_vec_size, self.value_vec_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            values_t = tf.transpose(values, perm=[0, 2, 1])  # (batch_size, value_vec_size, num_values)
            '''
            attn_logits = tf.matmul(keys, values_t)  # shape (batch_size, num_keys, num_values)
            '''
            # the problem here is that tf.scan only works when part_logits is of the same size as part_logits.
            part_logits = tf.scan(lambda a, x: tf.matmul(x, W), keys)
            attn_logits = tf.matmul(part_logits, values_t)
            attn_logits_mask = tf.expand_dims(values_mask, 1)  # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask,
                                          2)  # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values)  # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDAF(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        self.num_values = values.get_shape()[1]  # M
        self.num_keys = keys.get_shape()[1]  # N
        assert (self.key_vec_size == self.value_vec_size)
        with vs.variable_scope("BiDAF"):
            values_exp = tf.expand_dims(values, 1)  # question, (bc, 1, M, 2h)
            values_exp = tf.tile(values_exp, [1, self.num_keys, 1, 1])  # (bc, N, M, 2h)
            values_mask_exp = tf.tile(tf.expand_dims(values_mask, -2), [1, self.num_keys, 1])  # bc,M -> bc, N, M

            keys_exp = tf.expand_dims(keys, 2)  # context, (batch_size, N, 1, 2h)
            keys_exp = tf.tile(keys_exp, [1, 1, self.num_values, 1])  # (bc, N, M, 2h)
            keys_mask_exp = tf.tile(tf.expand_dims(keys_mask, -1), [1, 1, self.num_values])  # (bc, N, M)

            kv_mask = tf.cast(keys_mask_exp, tf.bool) & tf.cast(values_mask_exp, tf.bool)
            ######### self implementation #########
            cat_data = tf.concat([keys_exp, values_exp, tf.multiply(values_exp, keys_exp)], axis=3)  # (bc, N, M, 6h)
            assert cat_data.get_shape()[3] == 3 * self.key_vec_size, "BiDAF err: shape of cat_data is wrong"

            w_sim = tf.get_variable('w_sim', shape=[3 * self.key_vec_size, ], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
            s = tf.multiply(cat_data, w_sim)
            s = tf.reduce_sum(tf.multiply(cat_data, w_sim), axis=3)
            s = tf.nn.dropout(s, self.keep_prob)
            ########## imported lib #####
            # s = utils.get_logits([keys_exp, values_exp], None, True, is_train=(self.keep_prob < 1.0), func='tri_linear', input_keep_prob = self.keep_prob)
            ##############################
            print "similarity matrix: ", s.get_shape()  # s : (bc, N, M)
            assert (s.get_shape().ndims == kv_mask.get_shape().ndims)
            s_exp_masked, alpha = masked_softmax(s, kv_mask, 2)

            alpha = tf.reshape(alpha, shape=[-1, self.num_keys, self.num_values, 1])
            val_aug = tf.reshape(values, shape=[-1, 1, self.num_values, self.value_vec_size])
            val_a = tf.reduce_sum(tf.multiply(alpha, val_aug), axis=-2)  # bc, N, 2h

            beta = tf.nn.softmax(tf.reduce_max(s_exp_masked, axis=-1), dim=-1)
            beta = tf.reshape(beta, shape=[-1, self.num_keys, 1])
            key_aug = tf.reshape(keys, shape=[-1, self.num_keys, self.key_vec_size])
            key_a = tf.reduce_sum(tf.multiply(beta, key_aug), axis=-2)
            assert key_a.get_shape().as_list() == [None, self.key_vec_size]

            key_a = tf.tile(tf.expand_dims(key_a, -2), [1, self.num_keys, 1])  # bc, N, 2h

            key_val_a = keys * val_a  # bc, N, 2h
            key_key_a = keys * key_a  # bc, N, 2h
            out = tf.concat([val_a, key_val_a, key_key_a], axis=2)
            return out

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask)  # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist