#!/usr/bin/env python
# encoding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from mt_ie import data_utils



class Decoder(object):

    def __init__(self, sess, model, source_vocab, rev_target_vocab, beam_size):
        self.sess = sess
        self.model = model
        self.source_vocab = source_vocab
        self.rev_target_vocab = rev_target_vocab
        self.beam_size = beam_size

    def __enter__(self):
        return self

    def _decode_many(self, data_set):
        pair_ids, references, hypotheses = [], [], []

        start_from, step = 0, 0
        while start_from != len(data_set):
            # Decode.
            (encoder_inputs, decoder_inputs, target_weights,
             encoder_input_length, list_of_mask, batch_size,
             batch_references, start_from) = self.model.get_batch(data_set, 1, start_from)
            _, _, output_logits =  self.model.step(
                self.sess, encoder_inputs,decoder_inputs,
                target_weights, encoder_input_length,
                list_of_mask, batch_size, True)
            if self.beam_size > 1:
                # Beam search.
                model_outputs = Decoder.beam_decode(
                    output_logits[0], output_logits[1], output_logits[2], self.rev_target_vocab)
            else:
                model_outputs = [[] for i in xrange(batch_size)]
                for step_outputs in output_logits:
                    # Always choose the most probable.
                    decoded_token_ids = map(int, list(np.argmax(step_outputs, axis=1)))
                    for batch_idx in xrange(batch_size):
                        model_outputs[batch_idx].append(decoded_token_ids[batch_idx])
            # Truncate EOS and replace ids with tokens.
            for batch_idx in xrange(batch_size):
                pair_id, gold_outputs, has_ukn = batch_references[batch_idx]
                x = model_outputs[batch_idx][0] if self.beam_size > 1 else model_outputs[batch_idx]
                if data_utils.EOS_ID in x:
                    x = x[:x.index(data_utils.EOS_ID)]
                x = [tf.compat.as_str(self.rev_target_vocab[output]) for output in x]
                y = [tf.compat.as_str(self.rev_target_vocab[output])
                     for output in gold_outputs[:-1]]
                pair_ids.append(int(pair_id))
                references.append(" ".join(y).encode('utf8'))
                hypotheses.append(" ".join(x).encode('utf8'))
            step += 1
            if step % 10 == 0:
                print("=", end="")
                sys.stdout.flush()
        return pair_ids, references, hypotheses


    def __call__(self, source_sent):
        # Get token-ids for the input sentence.
        token_ids, has_ukn, ukn_tokens = data_utils.sentence_to_token_ids(
            tf.compat.as_bytes(source_sent), self.source_vocab)
        # Get a 1-element batch to feed the sentence to the model.
        (encoder_inputs, decoder_inputs, target_weights, encoder_input_length,
         list_of_mask, batch_size, _, _)  = self.model.get_batch(
            [(token_ids, [], 1)], 1)
        # Get output logits for the sentence.
        _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                         target_weights, encoder_input_length,
                                         list_of_mask, batch_size, True)
        outputs = []
        if self.beam_size > 1:
            # A beam decoder
            ret = []
            model_outputs = Decoder.beam_decode(
                output_logits[0], output_logits[1], output_logits[2], self.rev_target_vocab)
            outputs = model_outputs[0][0]
            for outputs in model_outputs[0]:
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                ret.append(" ".join([tf.compat.as_str(self.rev_target_vocab[output]) for output in outputs]))
            return ret
        else:
            # A greedy decoder
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        return (" ".join([tf.compat.as_str(self.rev_target_vocab[output]) for output in outputs]))

    def __exit__(self, exc_type, exc_value, trace_back):
        self.sess.close()

    @staticmethod
    def beam_decode(beam_symbols, beam_path, beam_log_probs, rev_target_vocab):
        step = 0
        while len(beam_symbols) != 0:
            if step == 0:
                current_symbols = beam_symbols.pop()
                current_symbols_parents = beam_path.pop()
                batch_size, beam_size = current_symbols.shape
                batch_sequences = [[[] for x in xrange(beam_size)] for y in xrange(batch_size)]
            else:
                # get parents
                prev_beam_symbols = beam_symbols.pop()
                prev_beam_parents = beam_path.pop()
                temp_symbols, temp_symbols_parents = [], []
                for batch_idx in xrange(batch_size):
                    indices = current_symbols_parents[batch_idx]
                    symbols = prev_beam_symbols[batch_idx]
                    parents = prev_beam_parents[batch_idx]
                    temp_symbols.append(symbols[indices])
                    temp_symbols_parents.append(parents[indices])
                current_symbols, current_symbols_parents = temp_symbols, temp_symbols_parents
            step += 1
            # add symbols
            for batch_idx, step_symbols in enumerate(current_symbols):
                for beam_idx, symbol in enumerate(step_symbols):
                    batch_sequences[batch_idx][beam_idx].insert(0, symbol)
        return batch_sequences

    @staticmethod
    def get_instance(sess, model, FLAGS):
        # Load vocabularies.
        source_vocab_path = os.path.join(
            FLAGS.data_dir,"vocab-%d.%s" % (FLAGS.vocab_size, FLAGS.source_language))
        target_vocab_path = os.path.join(
            FLAGS.data_dir, "vocab-%d.%s" % (FLAGS.vocab_size, FLAGS.target_language))
        source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
        _, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)

        return Decoder(sess, model, source_vocab, rev_target_vocab, FLAGS.beam_size)



