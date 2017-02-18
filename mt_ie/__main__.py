#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import time
import math
import glob
import shutil

import numpy as np
import tensorflow as tf

from mt_ie import seq2seq_model
from mt_ie import data_utils
from mt_ie import utils
from mt_ie.decoder import Decoder


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "Size of chinese vocabulary.")
tf.app.flags.DEFINE_integer("encoder_length", 50, "Encoder length.")
tf.app.flags.DEFINE_integer("decoder_length", 50, "decoder length.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_buckets", 10, "Number of buckets.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers for each cell.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each layer.")
tf.app.flags.DEFINE_integer("beam_size", 1, "Beam size for beam search (only used in evaluation).")
tf.app.flags.DEFINE_integer("num_samples", 1024, "Number of samples for sampled softmax.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("source_language", "zh", "Source language.")
tf.app.flags.DEFINE_string("target_language", "pp", "Target language.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory.")
tf.app.flags.DEFINE_string("model_dir", "./model", "Model directory.")
tf.app.flags.DEFINE_string("utils_dir", "./utils", "Model directory.")
tf.app.flags.DEFINE_boolean("use_dropout", False, "Using dropout at outputs of decoder cells.")
tf.app.flags.DEFINE_boolean("use_adam_opt", True, "Using Adam Optimizer.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Training using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("do_decode", False, "Flag for decoding disables backprop.")

FLAGS = tf.app.flags.FLAGS


def get_data_path():
    corpus_path = os.path.join(FLAGS.data_dir, "corpus")
    source_train = corpus_path + (".ids-%d.train.%s"
                                  % (FLAGS.vocab_size, FLAGS.source_language))
    target_train = corpus_path + (".ids-%d.train.%s"
                                  % (FLAGS.vocab_size, FLAGS.target_language))
    source_dev = corpus_path + (".ids-%d.test.%s"
                                % (FLAGS.vocab_size, FLAGS.source_language))
    target_dev = corpus_path + (".ids-%d.test.%s"
                                % (FLAGS.vocab_size, FLAGS.target_language))
    return source_train, target_train, source_dev, target_dev


def get_vocab_size(FLAGS=FLAGS):
    source_vocab_path = os.path.join(
        FLAGS.data_dir, "vocab-%d.%s" % (FLAGS.vocab_size, FLAGS.source_language))
    target_vocab_path = os.path.join(
        FLAGS.data_dir, "vocab-%d.%s" % (FLAGS.vocab_size, FLAGS.target_language))
    return utils.wc(source_vocab_path), utils.wc(target_vocab_path)


def create_model(sess, FLAGS=FLAGS):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    source_vocab_size, target_vocab_size = get_vocab_size(FLAGS)
    model = seq2seq_model.Seq2SeqModel(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        encoder_length=FLAGS.encoder_length,
        decoder_length=FLAGS.decoder_length,
        size=FLAGS.size,
        num_layers=FLAGS.num_layers,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        beam_size=FLAGS.beam_size,
        num_samples=FLAGS.num_samples,
        use_dropout=FLAGS.use_dropout,
        use_adam_opt=FLAGS.use_adam_opt,
        do_decode=FLAGS.do_decode,
        dtype=dtype)
    print("FLAGS.model_dir: %s" % FLAGS.model_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Create model with fresh parameters")
        sess.run(tf.global_variables_initializer())
    print("Vocab size: (%d, %d)" %(source_vocab_size, target_vocab_size))

    return model


def get_funct_for_next_bucket(train_set, num_buckets):
    if num_buckets > 1:
        buckets = data_utils.bucket(train_set, num_buckets)
        train_bucket_sizes = [len(buckets[bucket_id]) for bucket_id in xrange(num_buckets)]
        train_total_size = sum(train_bucket_sizes)
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in xrange(len(train_bucket_sizes))]
    def get_next_bucket():
        if num_buckets > 1:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            return bucket_id, buckets[bucket_id]
        else:
            return 0, train_set
    return get_next_bucket


def train():
    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess)

        print("Reading data.")
        source_train, target_train, source_dev, target_dev = get_data_path()
        _train_set = data_utils.read_data(source_train, target_train,
                                         FLAGS.encoder_length, FLAGS.decoder_length)
        # Bucket it.
        get_next_bucket = get_funct_for_next_bucket(_train_set, FLAGS.num_buckets)
        dev_set = data_utils.read_data(source_dev, target_dev,
                                       FLAGS.encoder_length, FLAGS.decoder_length)

        # Training step.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        data_pos = None   # Random sample.
        while True:
            bucket_id, train_set = get_next_bucket()
            # Get a batch and make a step.
            start_time = time.time()
            (encoder_inputs, decoder_inputs, target_weights,
             encoder_input_length, list_of_mask, batch_size,
             references, data_pos) = model.get_batch(train_set, 0, data_pos)
            _, step_loss, _ =  model.step(sess, encoder_inputs, decoder_inputs,
                                          target_weights, encoder_input_length,
                                          list_of_mask, batch_size, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % (FLAGS.steps_per_checkpoint // 5) == 0:
                print("=", end="")
                sys.stdout.flush()


            # Once in a while, save checkpoint, print statistics.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                global_step = model.global_step.eval()
                learning_rate = model.learning_rate.eval()
                sample_count = data_pos if data_pos is not None else (global_step * FLAGS.batch_size)
                coverage = sample_count * 100 / len(_train_set)
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("=> | batch %8d | %6.2f%% epoch | lr %6.2f | %6.4f s/batch |"
                      " loss %6.2f | ppl %6.2f |" % (
                      global_step, coverage, learning_rate, step_time,
                      loss, perplexity), end="")
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint.
                checkpoint_path = os.path.join(FLAGS.model_dir, "mt-ie.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                dev_pos = -1
                dev_loss = 0
                dev_step = 0
                while dev_pos != len(dev_set):
                    (encoder_inputs, decoder_inputs, target_weights,
                     encoder_input_length, list_of_mask, batch_size,
                     references, dev_pos) = model.get_batch(dev_set, 0, dev_pos)
                    _, step_loss, _ =  model.step(sess, encoder_inputs, decoder_inputs,
                                                  target_weights, encoder_input_length,
                                                  list_of_mask, batch_size, True)
                    dev_loss += step_loss
                    dev_step += 1
                    if dev_step % 10 == 0:
                        print("=", end="")
                        sys.stdout.flush()
                dev_loss /= dev_step
                dev_ppl = math.exp(float(dev_loss)) if dev_loss < 300 else float("inf")
                print(" => | dev loss %6.2f | dev ppl %6.2f |" % (dev_loss, dev_ppl), end="")

                # Copy the best.
                val_best = model.val_best.eval()
                print(" val_best %6.2f |" % val_best)
                if val_best == 0.0 or val_best < dev_ppl:
                    # Clean the previous best.
                    val_best_dir = os.path.join(FLAGS.model_dir, "val_best")
                    shutil.rmtree(val_best_dir, ignore_errors=True)
                    os.mkdir(val_best_dir)
                    # Copy the new best.
                    model_path_pattern = checkpoint_path + "-%d*" % global_step
                    for filepath in glob.glob(model_path_pattern):
                        shutil.copy(filepath, val_best_dir)
                    # Add checkpoint file.
                    ckpt_list_path = os.path.join(val_best_dir, "checkpoint")
                    val_best_ckpt_path = (os.path.join(val_best_dir, "mt-ie.ckpt")
                                          + "-%d" % global_step)
                    with open(ckpt_list_path, "w") as ckpt:
                        ckpt.write("model_checkpoint_path: \"%s\"\n"
                                   "all_model_checkpoint_paths: \"%s\""
                                   % (val_best_ckpt_path, val_best_ckpt_path))
                    # Record the best ppl.
                    assign_op = model.val_best.assign(tf.constant(dev_ppl))
                    sess.run(assign_op)

                sys.stdout.flush()


def evaluate():
    corpus_path = os.path.join(FLAGS.data_dir, "corpus")
    source_test = corpus_path + (".ids-%d.test.%s"
                                 % (FLAGS.vocab_size, FLAGS.source_language))
    target_test = corpus_path + (".ids-%d.test.%s"
                                 % (FLAGS.vocab_size, FLAGS.target_language))
    test_set = data_utils.read_data(source_test, target_test,
                                   FLAGS.encoder_length, FLAGS.decoder_length)
    with tf.Session() as sess:
        model = create_model(sess, FLAGS)
        decoder = Decoder.get_instance(sess, model, FLAGS)
        pair_ids, references, hypotheses = decoder._decode_many(test_set)

    bleu_score, status = utils.bleu_score(hypotheses, references, FLAGS.utils_dir)

    print("====> BLEU: %.4f (%s)" % (bleu_score, status))


def main(_):
    if FLAGS.do_decode:
        evaluate()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()

