#!/usr/bin/env python3

# The main LemmaTag code can be found below.
# For an up-to-date version of this code, see https://github.com/Hyperparticle/LemmaTag
# Author: Daniel Kondratyuk, Tomáš Gavenčiak, and Milan Straka

import numpy as np
import tensorflow as tf
import argparse
import datetime
import os
import shutil
import sys
from tqdm import tqdm
from tensorflow.python.client import timeline
import logging
from logging import warning, info, debug, error

from util import morpho_dataset
from util.utils import MorphoAnalyzer, Tee, log_time, find_first, AddInputsWrapper
from util.tags import WholeTags, CharTags, DictTags
from model.encoder import encoder_network
from model.tag_decoder import tag_decoder, tag_features
from model.lemma_decoder import lemma_decoder, sense_predictor


class LemmaTagNetwork:
    """LemmaTag class that constructs, trains, and evaluates the model"""

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, lem_num_chars, num_tags, num_senses, bow, eow):
        with self.session.graph.as_default():
            # Training params
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

            # Sentence lengths
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            # Number of output words
            self.words_count = tf.reduce_sum(self.sentence_lens)
            words_count = self.words_count
            # Map sentences -> word list
            self.word_indexes = tf.placeholder(tf.int32, [None, 2], name='word_indexes')

            # Tag data
            self.tags = tf.placeholder(tf.int32, [None, None, len(num_tags)], name="tags")

            # Form IDs and charseqs
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")

            # Lemma charseqs
            self.target_senses = tf.placeholder(tf.int32, [None, None], name="target_senses")
            self.target_ids = tf.placeholder(tf.int32, [None, None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")

            # Sentence weights
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            sum_weights = tf.reduce_sum(weights)

            # Source forms lengths (in sentences and by words/lemmas)
            sentence_form_len = tf.nn.embedding_lookup(self.charseq_lens, self.charseq_ids)
            word_form_len = tf.gather_nd(sentence_form_len, self.word_indexes)

            # Target sequences for words
            _target_seq_lens = tf.nn.embedding_lookup(self.target_seq_lens, self.target_ids) # 2D
            _target_seqs = tf.nn.embedding_lookup(self.target_seqs, self.target_ids)
            # Flattened to word-list
            target_lens = tf.gather_nd(_target_seq_lens, self.word_indexes)
            target_seqs = tf.gather_nd(_target_seqs, self.word_indexes)
            target_senses = tf.gather_nd(self.target_senses, self.word_indexes)
            # Add eow at the end
            target_seqs = tf.reverse_sequence(target_seqs, target_lens, 1)
            target_seqs = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=eow)
            target_lens = target_lens + 1
            target_seqs = tf.reverse_sequence(target_seqs, target_lens, 1)

            # RNN Cell
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Encoder
            enc_out = encoder_network(self.word_indexes, self.word_ids, self.charseqs, self.charseq_ids,
                                      self.charseq_lens, self.sentence_lens, num_words, num_chars, args.we_dim,
                                      args.cle_dim, rnn_cell, args.rnn_cell_dim, args.rnn_layers, args.dropout,
                                      self.is_training, args.separate_embed, args.separate_rnn)
            rnn_inputs_tags, word_rnn_outputs, sentence_rnn_outputs_tags, word_cle_states, word_cle_outputs = enc_out

            # Tagger
            loss_tag, tag_outputs, self.predictions, correct_tag, correct_tags_compositional = tag_decoder(
                self.tags, sentence_rnn_outputs_tags, weights, sum_weights, num_tags, args.tags, args.label_smoothing)

            # Tagger features for lemmatizer
            tag_feats = tag_features(tag_outputs, self.word_indexes, words_count, args.rnn_cell_dim, args.dropout,
                                        self.is_training, args.no_tags_to_lemmas, args.tag_signal_dropout)

            self.current_accuracy_tag, self.update_accuracy_tag = tf.metrics.mean(correct_tag, weights=sum_weights)
            self.current_accuracy_tags_compositional, self.update_accuracy_tags_compositional = tf.metrics.mean(
                correct_tags_compositional)

            # Lemmatizer
            loss_lem, predictions = lemma_decoder(word_rnn_outputs, tag_feats, word_cle_states, word_cle_outputs,
                                                  word_form_len, target_seqs, target_lens, self.charseq_lens,
                                                  words_count, lem_num_chars, rnn_cell, args.rnn_cell,
                                                  args.rnn_cell_dim, args.cle_dim, args.beams, args.beam_len_penalty,
                                                  args.lem_smoothing, bow, eow)
            self.lemma_predictions_training, self.lemma_predictions, self.lemma_prediction_lengths = predictions

            # Lemmatizer sense predictor
            loss_sense, self.sense_prediction = sense_predictor(word_rnn_outputs, tag_feats, target_senses, num_senses,
                                                                words_count, args.predict_sense, args.sense_smoothing)

            # Lemma predictions, loss and accuracy
            self._lemma_stats(target_seqs, target_lens, target_senses)

            # Loss, training and gradients
            # Compute combined weighted loss on tags and lemmas
            loss = loss_tag + loss_lem * args.loss_lem_w + loss_sense * args.loss_sense_w
            self.global_step = tf.train.create_global_step()
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2)
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                self.gradient_norm = tf.global_norm(gradients)
                if args.grad_clip:
                    gradients, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
                self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step, name="training")

            # Saver
            self.saver = tf.train.Saver(max_to_keep=2)

            # Summaries
            self.current_loss_tag, self.update_loss_tag = tf.metrics.mean(loss_tag, weights=sum_weights)
            self.current_loss_lem, self.update_loss_lem = tf.metrics.mean(loss_lem, weights=sum_weights)
            self.current_loss_sense, self.update_loss_sense = tf.metrics.mean(loss_sense, weights=sum_weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=sum_weights)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=1 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(1):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss_tag", self.update_loss_tag),
                                           tf.contrib.summary.scalar("train/loss_sense", self.update_loss_sense),
                                           tf.contrib.summary.scalar("train/loss_lem", self.update_loss_lem),
                                           tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient", self.gradient_norm),
                                           tf.contrib.summary.scalar("train/accuracy_tag", self.update_accuracy_tag),
                                           tf.contrib.summary.scalar("train/accuracy_compositional_tags", self.update_accuracy_tags_compositional),
                                           tf.contrib.summary.scalar("train/accuracy_lem", self.update_accuracy_lem_train),
                                           tf.contrib.summary.scalar("train/accuracy_lemsense", self.update_accuracy_lemsense_train),
                                           tf.contrib.summary.scalar("train/learning_rate", self.learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy_tag", self.current_accuracy_tag),
                                               tf.contrib.summary.scalar(dataset + "/accuracy_compositional_tags", self.current_accuracy_tags_compositional),
                                               tf.contrib.summary.scalar(dataset + "/accuracy_lem", self.current_accuracy_lem),
                                               tf.contrib.summary.scalar(dataset + "/accuracy_lemsense", self.current_accuracy_lemsense)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def _lemma_stats(self, target_seqs, target_lens, target_senses):
        # Training accuracy
        accuracy_training_lem = tf.reduce_all(tf.logical_or(
            tf.equal(self.lemma_predictions_training, target_seqs),
            tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
        self.current_accuracy_lem_train, self.update_accuracy_lem_train = tf.metrics.mean(accuracy_training_lem)
        accuracy_training_lemsense = tf.logical_and(
            accuracy_training_lem,
            tf.equal(self.sense_prediction, tf.cast(target_senses, dtype=tf.int64)))
        self.current_accuracy_lemsense_train, self.update_accuracy_lemsense_train = tf.metrics.mean(
            accuracy_training_lemsense)

        # Predict accuracy
        minimum_length = tf.minimum(tf.shape(self.lemma_predictions)[1], tf.shape(target_seqs)[1])
        correct_lem = tf.logical_and(
            tf.equal(self.lemma_prediction_lengths, target_lens),
            tf.reduce_all(tf.logical_or(
                tf.equal(self.lemma_predictions[:, :minimum_length], target_seqs[:, :minimum_length]),
                tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
        self.current_accuracy_lem, self.update_accuracy_lem = tf.metrics.mean(correct_lem)
        correct_lemsense = tf.logical_and(
            correct_lem,
            tf.equal(self.sense_prediction, tf.cast(target_senses, dtype=tf.int64)))
        self.current_accuracy_lemsense, self.update_accuracy_lemsense = tf.metrics.mean(correct_lemsense)

    def train_epoch(self, train, args, rate):
        first = True
        with tqdm(total=len(train.sentence_lens), file=args.realstderr, unit="sent") as progress_bar:
            while not train.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes = train.next_batch(args.batch_size, including_charseqs=True)
                if args.word_dropout:
                    mask = np.random.binomial(n=1, p=args.word_dropout, size=word_ids[train.FORMS].shape)
                    word_ids[train.FORMS] = (1 - mask) * word_ids[train.FORMS] + mask * train.factors[train.FORMS].words_map["<unk>"]

                self.session.run(self.reset_metrics)
                # Chrome tracing graph
                if args.record_trace and first:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    options = None
                    run_metadata = None
                self.session.run([self.training, self.summaries["train"]],
                                 {self.sentence_lens: sentence_lens, self.learning_rate: rate,
                                  self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                  self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                                  self.target_ids: charseq_ids[train.LEMMAS], self.target_seqs: charseqs[train.LEMMAS],
                                  self.target_seq_lens: charseq_lens[train.LEMMAS], self.target_senses: word_ids[train.SENSES],
                                  self.tags: args.tags.encode(word_ids[train.TAGS], charseq_ids[train.TAGS], charseqs[train.TAGS]),
                                  self.is_training: True, self.word_indexes: word_indexes},
                                 options=options, run_metadata=run_metadata)
                progress_bar.update(len(sentence_lens))
                if args.record_trace and first:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    gs = self.session.run(self.global_step)
                    with open(args.logdir + '/timeline_train_{}.json'.format(gs), 'w', encoding="utf-8") as f:
                        f.write(chrome_trace)
                first = False

    def evaluate(self, dataset_name, dataset, args):
        self.session.run(self.reset_metrics)
        with tqdm(total=len(dataset.sentence_lens), file=args.realstderr, unit="sent") as progress_bar:
            while not dataset.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes = dataset.next_batch(args.batch_size, including_charseqs=True)
                self.session.run([self.update_accuracy_tag, self.update_accuracy_tags_compositional, self.update_accuracy_lem, self.update_accuracy_lemsense, self.update_loss],
                                 {self.sentence_lens: sentence_lens,
                                  self.charseqs: charseqs[dataset.FORMS], self.charseq_lens: charseq_lens[dataset.FORMS],
                                  self.word_ids: word_ids[dataset.FORMS], self.charseq_ids: charseq_ids[dataset.FORMS],
                                  self.target_ids: charseq_ids[dataset.LEMMAS], self.target_seqs: charseqs[dataset.LEMMAS],
                                  self.target_seq_lens: charseq_lens[dataset.LEMMAS], self.target_senses: word_ids[dataset.SENSES],
                                  self.tags: args.tags.encode(word_ids[dataset.TAGS], charseq_ids[dataset.TAGS], charseqs[dataset.TAGS]),
                                  self.is_training: False, self.word_indexes: word_indexes})
                progress_bar.update(len(sentence_lens))
        return self.session.run([self.current_accuracy_tag, self.current_accuracy_lem, self.current_accuracy_lemsense] + self.summaries[dataset_name])[:3]

    def predict(self, dataset, args):
        tags = []
        lemmas = []
        alphabet = dataset.factors[dataset.LEMMAS].alphabet
        sense_words = dataset.factors[dataset.SENSES].words
        with tqdm(total=len(dataset.sentence_lens), file=args.realstderr, unit="sent") as progress_bar:
            while not dataset.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes = dataset.next_batch(args.batch_size, including_charseqs=True)
                tp, lp, lpl, senses = self.session.run(
                    [self.predictions, self.lemma_predictions, self.lemma_prediction_lengths, self.sense_prediction],
                    {self.sentence_lens: sentence_lens,
                     self.charseqs: charseqs[dataset.FORMS], self.charseq_lens: charseq_lens[dataset.FORMS],
                     self.word_ids: word_ids[dataset.FORMS], self.charseq_ids: charseq_ids[dataset.FORMS],
                     self.is_training: False, self.word_indexes: word_indexes})
                tags.extend(args.tags.decode(tp))
                for si, length in enumerate(sentence_lens):
                    lemmas.append([])
                    for i in range(length):
                        lemmas[-1].append(''.join(alphabet[lp[i][j]] for j in range(lpl[i] - 1)))
                        if args.predict_sense:
                            if senses[i] > 0 and sense_words[senses[i]]:
                                sword = sense_words[senses[i]]
                                if sword and sword != "<pad>":
                                    lemmas[-1][-1] += "-{}".format(sword)
                    lp, lpl, senses = lp[length:], lpl[length:], senses[length:]
                assert len(lpl) == 0
                progress_bar.update(len(sentence_lens))

        return lemmas, tags


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # General and training arguments
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--name", default="", type=str, help="Any name comment.")
    parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint restore directory.")
    parser.add_argument("--beta_2", default=0.99, type=float, help="Adam beta 2.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--drop_rate_after", default=20, type=int, help="Number of epochs after which the rate is quartered every 10 epochs.")
    parser.add_argument("--grad_clip", default=3.0, type=float, help="Gradient clipping (if set).")
    parser.add_argument("--record_trace", default=False, action="store_true", help="Record training trace as Chrome trace (load at 'chrome://tracing/').")
    parser.add_argument("--no_save_net", default=False, action="store_true", help="Skip checkoint saving (to save space when debugging).")
    parser.add_argument("--only_eval", default=False, action="store_true", help="Skip training and only evaluate once (from a checkpoint).")

    # Data and seed
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--train", default="data/sample-cs-cltt-ud-train.txt", type=str, help="Training data path.")
    parser.add_argument("--dev", default="data/sample-cs-cltt-ud-dev.txt", type=str, help="Validation data path.")
    parser.add_argument("--test", default="data/sample-cs-cltt-ud-test.txt", type=str, help="Test data path.")
    parser.add_argument("--conllu", default=False, action="store_true", help="Using a conllu-formatted dataset")
    parser.add_argument("--analyzer", default=None, type=str, help="Analyzer text file (default none).")
    parser.add_argument("--max_sentences", default=None, type=int, help="Max sentences to load (for quick testing).")

    # Dimensions and features
    parser.add_argument("--cle_dim", default=64, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=2, type=int, help="RNN layers.")
    parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
    parser.add_argument("--att_dim", default=64, type=int, help="Attention dimension.")
    parser.add_argument("--predict_sense", default=False, action="store_true", help="Train and predict the sense as a part of the lemma (use dataset with separated sense for that).")
    parser.add_argument("--no_tags_to_lemmas", default=False, action="store_true", help="Don't use tag components as a signal to the lemmatizer.")
    parser.add_argument("--separate_rnn", default=False, action="store_true", help="Use separate RNN for tags and lemmas/senses.")
    parser.add_argument("--separate_embed", default=False, action="store_true", help="Use separate embeddings for tags and lemmas/senses. Implies separate_rnn.")
    parser.add_argument("--beams", default=None, type=int, help="Use beam search with the given no of beams.")
    parser.add_argument("--loss_sense_w", default=0.1, type=float, help="Sense loss weight (if sense is separate).")
    parser.add_argument("--loss_lem_w", default=1.0, type=float, help="Lemmatization loss weight.")

    # Regularization
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--lem_smoothing", default=0.0, type=float, help="Lemma label smoothing.")
    parser.add_argument("--sense_smoothing", default=0.05, type=float, help="Sense label smoothing.")
    parser.add_argument("--word_dropout", default=0.25, type=float, help="Word dropout")
    parser.add_argument("--tag_signal_dropout", default=None, type=float, help="Tag signal dropout to lemmatizer")
    parser.add_argument("--tag_type", default="char", choices=["char", "dict", "whole"], help="Compositional tag type.")
    parser.add_argument("--compositional_tags_regularization", default=0.1, type=float, help="Compositional tags regularization.")
    parser.add_argument("--whole_tags_regularization", default=1.0, type=float, help="Whole tags regularization.")
    parser.add_argument("--beam_len_penalty", default=0.2, type=float, help="BeamSearch length_penalty_weight param.")

    args = parser.parse_args()
    if args.separate_embed:
        args.separate_rnn = True
    if args.only_eval:
        args.epochs = 1

    # Fix the random seed
    np.random.seed(args.seed)

    # Logdir, copy the source and log the outputs
    if not os.path.exists("logs"): os.mkdir("logs")
    basename = "LT-{}-{}-S{}".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        args.name, args.seed, )
    args.logdir = "logs/" + basename
    os.mkdir(args.logdir)
    shutil.copy(__file__, args.logdir + "/taglem.py")
    tee = Tee(args.logdir + "/log.txt")
    tee.start()
    args.realstderr = tee.stderr # A bit hacky ... (to leave progress bars out of logs)
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.DEBUG)

    info("Running in {} with args: {}".format(args.logdir, str(args)))
    info("Commandline: {}".format(' '.join(sys.argv)))

    # Load the data
    with log_time("load inputs"):
        args.max_dev_sentences = args.max_sentences // 5 if args.max_sentences else None
        train = morpho_dataset.MorphoDataset(args.train, max_sentences=args.max_sentences, conllu_format=args.conllu)
        dev = morpho_dataset.MorphoDataset(args.dev, train=train, shuffle_batches=False, max_sentences=args.max_dev_sentences, conllu_format=args.conllu)
        test = morpho_dataset.MorphoDataset(args.test, train=train, shuffle_batches=False, max_sentences=args.max_dev_sentences, conllu_format=args.conllu)
        # analyzer = MorphoAnalyzer(args.analyzer) if args.analyzer else None

    # Construct the network
    if args.tag_type == "char":
        args.tags = CharTags(train, args.compositional_tags_regularization, args.whole_tags_regularization)
    elif args.tag_type == "dict":
        # TODO: add support for dictionary tags
        raise ValueError("Tag type not supported: " + args.tag_type)
        # args.tags = DictTags(train, args.compositional_tags_regularization, args.whole_tags_regularization)
    elif args.tag_type == "whole":
        args.tags = WholeTags(train)
    else:
        raise ValueError("Invalid tag_type")

    network = LemmaTagNetwork(threads=args.threads, seed=args.seed)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.LEMMAS].alphabet), args.tags.num_tags(),
                      len(train.factors[train.SENSES].words), train.factors[train.LEMMAS].alphabet_map["<bow>"],
                      train.factors[train.LEMMAS].alphabet_map["<eow>"])

    if args.checkpoint:
        network.saver.restore(network.session, args.checkpoint)

    # Train
    dev_best = 0
    for ep in range(args.epochs):
        rate = args.learning_rate
        if args.drop_rate_after and args.drop_rate_after <= ep:
            rate = args.learning_rate * 0.25 ** (1 + ((ep - args.drop_rate_after) // 10))

        if not args.only_eval:
            info("Training epoch %d with rate %f", ep, rate)
            network.train_epoch(train, args, rate=rate)

        info("Evaluating dev")
        dev_acc_tag, dev_acc_lem, dev_acc_lemsense = network.evaluate("dev", dev, args)
        info(".. epoch {} (step {}) dev accuracy: {:.2f} tag, {:.2f} lemma, {:.2f} lemma with sense".format(
            ep, network.session.run(network.global_step), 100 * dev_acc_tag, 100 * dev_acc_lem, 100 * dev_acc_lemsense))

        if dev_acc_tag + dev_acc_lemsense > dev_best or ep == args.epochs - 1:
            if not args.no_save_net and not args.only_eval: # To speed up testing / save disk space :)
                network.saver.save(network.session, "{}/checkpoint".format(args.logdir), global_step=network.global_step, write_meta_graph=False)

            for dset, name in [(dev, "dev"), (test, "test")]:
                fname = "{}/taglem_{}_ep{}.txt".format(args.logdir, name, ep)
                info("Predicting %s into %s", name, fname)
                with open(fname, "w", encoding="utf-8") as ofile:
                    forms = dset.factors[dset.FORMS].strings
                    lemmas, tags = network.predict(dset, args)
                    for s in range(len(forms)):
                        for i in range(len(forms[s])):
                            print("{}\t{}\t{}".format(forms[s][i], lemmas[s][i], tags[s][i]), file=ofile)
                        print("", file=ofile)

        dev_best = max(dev_best, dev_acc_tag + dev_acc_lemsense)
