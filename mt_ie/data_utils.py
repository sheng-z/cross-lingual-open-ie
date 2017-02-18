#!/usr/bin/env python
# encoding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import re
import random

import tensorflow as tf
from tensorflow.python.platform import gfile


# Special vocabulary symbols - we always put them at the begining.
_PAD = b"_PAD"          # Padding symbol
_GO = b"_GO"            # Start-decoding symbol
_EOS = b"_EOS"          # End-of-sentence symbo
_UNK = b"_UNK"          # Unknown token
_UNK_A = b"_UNK:a"      # Unknown argument token
_UNK_P = b"_UNK:p"      # Unknown predicate token
_UNK_AH = b"_UNK:a_h"   # Unknown argument head
_UNK_PH = b"_UNK:p_h"   # Unknown predicate head

_START_VOCAB1 = [_PAD, _GO, _EOS, _UNK]
_START_VOCAB2 = [_PAD, _GO, _EOS, _UNK_A, _UNK_P]
_START_VOCAB3 = [_PAD, _GO, _EOS, _UNK_A, _UNK_P, _UNK_AH, _UNK_PH]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expression used to tokenize
_DIGIT_RE = re.compile(br"\d")


def convert_data_to_text(source_path, target_path,
                         source_rev_vocab, target_rev_vocab,
                         source_size, target_size):
    bitext = {}
    with gfile.GFile(source_path, "r") as source_file:
        with gfile.GFile(target_path, "r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("Processed %d lines" % counter)
                    sys.stdout.flush()
                source_text = [tf.compat.as_str(source_rev_vocab[int(x)])
                               for x in source.strip().split()]
                target_text = [tf.compat.as_str(target_rev_vocab[int(x)])
                               for x in target.strip().split()]
                if len(source_text) <= source_size and len(target_text) <= target_size + 1:   # EOS
                    bitext[counter] = (source_text, target_text)
                else:
                    pass
                    # print("Too long sent zh %d  en %d" % (len(source_ids), len(target_ids)))
                source, target = source_file.readline(), target_file.readline()
    return bitext


def read_data(source_path, target_path, source_size, target_size):
    data_set = []
    with gfile.GFile(source_path, "r") as source_file:
        with gfile.GFile(target_path, "r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target:
                counter += 1
                if counter % 100000 == 0:
                    print("Processed %d lines" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.strip().split()]
                target_ids = [int(x) for x in target.strip().split()]
                target_ids.append(EOS_ID)
                if len(source_ids) <= source_size and len(target_ids) <= target_size + 1:   # EOS
                    data_set.append([source_ids, target_ids, counter])
                else:
                    pass
                    # print("Too long sent zh %d  en %d" % (len(source_ids), len(target_ids)))
                source, target = source_file.readline(), target_file.readline()
    return data_set


def bucket(data_set, num_buckets=4):
    num_data = len(data_set)
    quota = num_data // num_buckets
    buckets = [[] for _ in xrange(num_buckets)]
    data_set = sorted(data_set, key=lambda x: len(x[1]))
    for i in xrange(num_buckets):
        buckets[i] = data_set[i*quota:(i+1)*quota]
    buckets[num_buckets-1] += data_set[num_buckets*quota:]
    return buckets


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, "rb") as f:
            rev_vocab = [line.strip() for line in f]
        vocab = {x:y for (y, x) in enumerate(rev_vocab)}
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def token_to_token_id(w, vocabulary, language="pp", normalize_digits=True):
    if normalize_digits:
        w = re.sub(_DIGIT_RE, b"0", w)
    if w not in vocabulary:
        if language == "pp":
            _, label = w.rsplit(":", 1)
            w = ":".join((_UNK, label))
        else:
            w = _UNK
    return vocabulary[w]


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
    # ids = []
    words = sentence.strip().split()
    token_id = lambda w: token_to_token_id(w, vocabulary, "zh", normalize_digits)
    token_ids = [token_id(w) for w in words]
    has_ukn = False
    ukn_tokens = []
    for idx, tk_id in enumerate(token_ids):
        if tk_id in {UNK_ID}:
            has_ukn = True
            ukn_tokens.append(words[idx])
    return token_ids, has_ukn, ukn_tokens


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      language="pp", normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Converting data %s to token ids %s" % (data_path, target_path))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        token_id = lambda w: token_to_token_id(w, vocab, language, normalize_digits)
        with gfile.GFile(data_path, "rb") as data_file:
            with gfile.GFile(target_path, "wb") as target_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processd %d lines" % counter)
                    tokens = line.strip().split()
                    token_ids = [token_id(w) for w in tokens]
                    target_file.write(" ".join(map(str, token_ids)) + "\n")
        print("  processd %d lines" % counter)


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=40000,
                      language="pp", normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        _start_vocab = _START_VOCAB1
        if language == "pp":
            _start_vocab = _START_VOCAB3
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processd %d lines" % counter)
                tokens = line.strip().split()
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
        print("  processd %d lines" % counter)
        vocab_list = _start_vocab + sorted(vocab, key=vocab.get, reverse=True)

        freq_list = {i:vocab[word] for (i, word) in enumerate(sorted(vocab, key=vocab.get, reverse=True))}
        total_word = sum(freq_list.itervalues())
        print("Total number of words: %d" % total_word)
        aggr = 0
        for i, freq in freq_list.iteritems():
            if i > max_vocabulary_size:
                print("current vocab_size accounts %f%%" % (aggr/total_word*100))
                break
            aggr += freq

        print("Record %d words!" % len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, "wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def split_data(ids_path, corpus_path, zh, en, pp, fold=25):
    languages = data_suffixes = [zh, en, pp]
    line_idx_list = None

    for data_path, suffixes in [(ids_path, data_suffixes), (corpus_path, languages)]:
        for suffix in suffixes:
            with gfile.GFile(data_path + "." + suffix) as data_file:
                lines = data_file.readlines()
            if line_idx_list is None:
                print("Splitting data for Cross-lingual IE and Translation.")
                line_idx_list = list(xrange(len(lines)))
                random.shuffle(line_idx_list)
                one_fold_len = int(len(line_idx_list) / fold)
                test_idx_list = line_idx_list[:one_fold_len]
                dev_idx_list= line_idx_list[one_fold_len:one_fold_len * 2]
                train_idx_list = line_idx_list[one_fold_len * 2:]
                print("  %d train lines; %d dev lines; %d test lines."
                      % (len(train_idx_list), len(dev_idx_list), len(test_idx_list)))
            elif len(lines) != len(line_idx_list):
                raise ValueError("%s has different number of lines." % (data_path + suffix))
            with gfile.GFile(data_path + ".train." + suffix, "w") as train_file:
                for line_idx in train_idx_list:
                    train_file.write(lines[line_idx])
            with gfile.GFile(data_path + ".dev." + suffix, "w") as dev_file:
                for line_idx in dev_idx_list:
                    dev_file.write(lines[line_idx])
            with gfile.GFile(data_path + ".test." + suffix, "w") as test_file:
                for line_idx in test_idx_list:
                    test_file.write(lines[line_idx])


def prepare_data(data_dir, zh, en, pp, corpus_name, vocabulary_size=40000):
    corpus_path = os.path.join(data_dir, corpus_name)

    # Create source vocabulary
    zh_vocab_path = os.path.join(data_dir, "vocab-%d.%s" % (vocabulary_size, zh))
    create_vocabulary(zh_vocab_path, corpus_path + "." + zh, vocabulary_size, zh)

    # Create target vocabularies
    en_vocab_path = os.path.join(data_dir, "vocab-%d.%s" % (vocabulary_size, en))
    pp_vocab_path = os.path.join(data_dir, "vocab-%d.%s" % (vocabulary_size, pp))

    create_vocabulary(en_vocab_path, corpus_path + "." + en, vocabulary_size, en)
    create_vocabulary(pp_vocab_path, corpus_path + "." + pp, vocabulary_size, pp, False)

    # Create token ids.
    zh_ids_path = corpus_path + (".ids-%d.%s" % (vocabulary_size, zh))
    en_ids_path = corpus_path + (".ids-%d.%s" % (vocabulary_size, en))
    pp_ids_path = corpus_path + (".ids-%d.%s" % (vocabulary_size, pp))
    data_to_token_ids(corpus_path + "." + zh, zh_ids_path, zh_vocab_path, zh)
    data_to_token_ids(corpus_path + "." + en, en_ids_path, en_vocab_path, en)
    data_to_token_ids(corpus_path + "." + pp, pp_ids_path, pp_vocab_path, pp, False)

    split_data(corpus_path + (".ids-%d" % vocabulary_size), corpus_path, zh, en, pp)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("data_dir", "data", "")
    tf.app.flags.DEFINE_string("source_lang", "zh", "")
    tf.app.flags.DEFINE_string("target_lang", "en", "")
    tf.app.flags.DEFINE_string("predpa_lang", "pp", "")
    tf.app.flags.DEFINE_string("corpus_name", "corpus", "")
    tf.app.flags.DEFINE_integer("vocab_size", 40000, "")
    FLAGS = tf.app.flags.FLAGS


    prepare_data(FLAGS.data_dir,FLAGS.source_lang, FLAGS.target_lang,
                 FLAGS.predpa_lang, FLAGS.corpus_name, FLAGS.vocab_size)
