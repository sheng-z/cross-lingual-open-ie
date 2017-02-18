#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from mt_ie import seq2seq
from mt_ie import data_utils


class Seq2SeqModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 encoder_length,
                 decoder_length,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 beam_size=1,
                 num_samples=1024,
                 use_dropout=False,
                 use_adam_opt=False,
                 do_decode=False,
                 dtype=tf.float32):
        with tf.variable_scope("seq2seq_model_initialization"):
            self.source_vocab_size = source_vocab_size
            self.target_vocab_size = target_vocab_size
            self.encoder_length = encoder_length
            self.decoder_length = decoder_length + 1    # Include the EOS symbol.
            self.beam_size = beam_size
            self.batch_size = None
            self.learning_rate = tf.get_variable(
                "learning_rate", [],
                trainable=False,
                dtype=dtype,
                initializer=tf.constant_initializer(learning_rate))
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.get_variable(
                "global_step", [],
                trainable=False,
                initializer=tf.constant_initializer(0))
            self.val_best = tf.get_variable(
                "val_best", [],
                trainable=False,
                initializer=tf.constant_initializer(0.0))

            # Create an output projection for sampled softmax.
            output_projection = None
            softmax_loss_function = None
            if num_samples > 0 and num_samples < self.target_vocab_size:
                # The default variable initializer is uniform_unit_scaling_initializer.
                w = tf.get_variable("proj_w", [size, self.target_vocab_size],
                                    dtype=dtype)
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
                output_projection = (w, b)

                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    local_w_t = tf.cast(w_t, tf.float32)
                    local_b = tf.cast(b, tf.float32)
                    local_inputs = tf.cast(inputs, tf.float32)
                    return tf.cast(
                        tf.nn.sampled_softmax_loss(local_w_t, local_b,
                                                   local_inputs, labels,
                                                   num_samples, self.target_vocab_size),
                        dtype)
                softmax_loss_function = sampled_loss

            # Define cells for encoder and decoder.
            encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            if use_dropout:
                encoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                    encoder_cell, output_keep_prob=0.5, input_keep_prob=0.5)
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                    decoder_cell, output_keep_prob=0.5, input_keep_prob=0.5)
            if num_layers > 1:
                encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * num_layers)
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_cell] * num_layers)

            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            self.list_of_mask = []
            for i in xrange(self.encoder_length):
                self.encoder_inputs.append(tf.placeholder(
                    tf.int32,
                    shape=[self.batch_size],
                    name="encoder{0}".format(i)))

            for i in xrange(self.decoder_length + 1):   # Include the Go symbol.
                # decoder_inputs = [Go, w_1, w_2, ..., w_n, EOS]
                self.decoder_inputs.append(tf.placeholder(
                    tf.int32,
                    shape=[self.batch_size],
                    name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(
                    dtype,
                    shape=[self.batch_size],
                    name="weight{0}".format(i)))
                self.list_of_mask.append(tf.placeholder(
                    tf.int32,
                    shape=[self.batch_size],
                    name="mask{0}".format(i)))

            # Sequence length for dynamic rnn.
            self.encoder_input_length = tf.placeholder(
                tf.int32,
                shape=[self.batch_size],
                name="encoder_input_length")

            # Targets are decoder inputs shifted by one.
            # targets = [w_1, w_2, ..., w_n, EOS]
            targets = [self.decoder_inputs[i + 1]
                       for i in xrange(len(self.decoder_inputs) - 1)]

            self.outputs, self.loss, self.beams = seq2seq.model(
                self.encoder_inputs[:self.encoder_length],
                self.decoder_inputs[:self.decoder_length],
                targets[:self.decoder_length],
                self.target_weights[:self.decoder_length],
                self.encoder_input_length,
                self.list_of_mask[:self.decoder_length],
                encoder_cell,
                decoder_cell,
                num_encoder_symbols=self.source_vocab_size,
                num_decoder_symbols=self.target_vocab_size,
                embedding_size=size,
                beam_size=beam_size,
                output_projection=output_projection,
                softmax_loss_function=softmax_loss_function,
                dtype=dtype)

            if output_projection is not None:
                self.projected_outputs = [
                    tf.matmul(output, output_projection[0]) + output_projection[1]
                    for output in self.outputs]
            else:
                self.projected_outputs = self.outputs

            if not do_decode:
                # Optimization for training the model.
                params = tf.trainable_variables()
                if use_adam_opt:
                    opt = tf.train.AdamOptimizer()
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.updates = opt.apply_gradients(zip(clipped_gradients, params),
                                                   global_step=self.global_step)

            self.saver = tf.train.Saver(tf.global_variables())


    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             encoder_input_length, list_of_mask, batch_size, do_decode=False):
        if len(encoder_inputs) != self.encoder_length:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), self.encoder_length))
        if len(decoder_inputs) != self.decoder_length:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), self.decoder_length))
        if len(target_weights) != self.decoder_length:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), self.decoder_length))

        input_feed = {}
        for l in xrange(self.encoder_length):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(self.decoder_length):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.list_of_mask[l].name] = list_of_mask[l]

        # Since targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[self.decoder_length].name
        input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

        input_feed[self.encoder_input_length.name] = encoder_input_length

        if not do_decode:
            output_feed = [self.updates,
                           self.gradient_norm,
                           self.loss]
        else:
            if self.beam_size > 1:
                output_feed = self.beams
            else:
                output_feed = [self.loss]
                for l in xrange(self.decoder_length):
                    output_feed.append(self.projected_outputs[l])

        outputs = session.run(output_feed, input_feed)
        if not do_decode:
            return outputs[1], outputs[2], None
        else:
            if self.beam_size > 1:
                return None, None, outputs
            return None, outputs[0], outputs[1:]


    def get_batch(self, data_set, feed_previous_rate=1, pos=-1):
        encoder_inputs, decoder_inputs = [], []
        encoder_input_length = []
        references = []

        for i in xrange(0, 64): # self.batch_size + 1):
            if pos is None:
                # random samples.
                encoder_input, decoder_input, pair_id = random.choice(data_set)
            else:
                pos += 1
                if pos != 0 and pos % len(data_set) == 0:
                    random.shuffle(data_set)
                    break
                encoder_input, decoder_input, pair_id = data_set[pos%len(data_set)]


            # Check if Unknown tokens are in the input
            has_ukn = True if data_utils.UNK_ID in encoder_input else False

            # Encoder inputs are padded.
            encoder_pad = [data_utils.PAD_ID] * (self.encoder_length - len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))

            # Record the meaningful encoder input length.
            encoder_input_length.append(len(encoder_input))

            # Decoder inputs get a starting symbol "GO", and are padded.
            decoder_pad = [data_utils.PAD_ID] * (self.decoder_length - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

            # Save references for evaluation.
            references.append([pair_id, decoder_input, has_ukn])

        encoder_input_length = np.array(encoder_input_length, dtype=np.int32)

        # batch_size is not necessarily equal to self.batch_size.
        batch_size = len(encoder_inputs)

        # Create the list of masks.
        list_of_mask = []
        full_matrix = np.full((batch_size), int(feed_previous_rate * 100))
        for length_idx in xrange(self.decoder_length):
            mask = np.greater(full_matrix,
                              np.random.randint(100, size=(batch_size))).astype(np.float32)
            list_of_mask.append(mask)


        # Now create time-major vectors from the data seleted above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(self.encoder_length):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        for length_idx in xrange(self.decoder_length):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in xrange(batch_size):
                if length_idx < self.decoder_length - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == self.decoder_length - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)



        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights,
                encoder_input_length, list_of_mask, batch_size, references,
                pos)

