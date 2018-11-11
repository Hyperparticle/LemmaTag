import tensorflow as tf
from util.utils import find_first, AddInputsWrapper


def lemma_decoder(word_rnn_outputs, tag_feats, word_cle_states, word_cle_outputs, word_form_len,
                  target_seqs, target_lens, charseq_lens, words_count, lem_num_chars, rnn_cell, rnn_cell_type,
                  rnn_cell_dim, cle_dim, beams, beam_len_penalty, lem_smoothing, bow, eow):
    """Decodes lemmas from a variety of inputs"""
    # Target embedding and target sequences
    tchar_emb = tf.get_variable('tchar_emb', [lem_num_chars, cle_dim])
    target_seqs_bow = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=bow)[:, :-1]
    tseq_emb = tf.nn.embedding_lookup(tchar_emb, target_seqs_bow)

    decoder_layer = tf.layers.Dense(lem_num_chars, name="decoder_layer")
    base_cell = rnn_cell(rnn_cell_dim, name="decoder_cell")

    def create_attn_cell(beams=None):
        with tf.variable_scope("lem_cell", reuse=tf.AUTO_REUSE):
            def btile(x):
                return tf.contrib.seq2seq.tile_batch(x, beams) if beams else x
            cell = base_cell
            cell = AddInputsWrapper(cell, btile(word_rnn_outputs))  # Already dropped out
            cell = AddInputsWrapper(cell, btile(tag_feats))         # Already dropped out
            cell = AddInputsWrapper(cell, btile(word_cle_states))
            att = tf.contrib.seq2seq.LuongAttention(rnn_cell_dim, btile(word_cle_outputs),
                                                    memory_sequence_length=btile(word_form_len))
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, att, output_attention=False)
            return cell

    train_cell = create_attn_cell()
    pred_cell = create_attn_cell(beams) # Reuses the attention memory

    if rnn_cell_type == "LSTM":
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(c=word_cle_states, h=word_cle_states)
    else:
        initial_state = word_cle_states

    # Training
    with tf.variable_scope("lem_decoder", reuse=tf.AUTO_REUSE):
        train_helper = tf.contrib.seq2seq.TrainingHelper(tseq_emb, sequence_length=target_lens, name="train_helper")
        train_initial_state = train_cell.zero_state(words_count, tf.float32).clone(cell_state=initial_state)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=train_cell, helper=train_helper, output_layer=decoder_layer, initial_state=train_initial_state)
        train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder)
        train_logits = train_outputs.rnn_output
        lemma_predictions_training = train_outputs.sample_id

    # Compute loss with smoothing
    with tf.variable_scope("lem_loss"):
        weights_reshaped = tf.reshape(tf.sequence_mask(target_lens, dtype=tf.float32), [-1])
        if lem_smoothing:
            train_logits_reshaped = tf.reshape(train_logits, [-1, train_logits.shape[-1]])
            gold_lemma_onehot = tf.one_hot(tf.reshape(target_seqs, [-1]), lem_num_chars)
            loss_lem = tf.losses.softmax_cross_entropy(gold_lemma_onehot,
                                                       train_logits_reshaped,
                                                       weights=weights_reshaped,
                                                       label_smoothing=lem_smoothing)
        else:
            loss_lem = tf.losses.sparse_softmax_cross_entropy(target_seqs, train_logits, weights=tf.sequence_mask(target_lens, dtype=tf.float32))

    # Predictions
    with tf.variable_scope("lem_decoder", reuse=tf.AUTO_REUSE):
        if not beams:
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=tchar_emb, start_tokens=tf.tile([bow], [words_count]), end_token=eow)
            pred_initial_state = pred_cell.zero_state(words_count, tf.float32).clone(cell_state=initial_state)
            pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell=pred_cell, helper=pred_helper, output_layer=decoder_layer, initial_state=pred_initial_state)
            pred_outputs, _, lemma_prediction_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=pred_decoder, maximum_iterations=tf.reduce_max(charseq_lens) + 10)
            lemma_predictions = tf.argmax(pred_outputs.rnn_output, axis=2, output_type=tf.int32)
        else:
            # Beam search predictions
            pred_initial_state = pred_cell.zero_state(words_count * beams, tf.float32).clone(cell_state=tf.contrib.seq2seq.tile_batch(initial_state, args.beams))
            pred_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                pred_cell, embedding=tchar_emb, start_tokens=tf.tile([bow], [words_count]),
                end_token=eow, output_layer=decoder_layer, beam_width=beams,
                initial_state=pred_initial_state, length_penalty_weight=beam_len_penalty)
            dec_outputs, dec_state, dec_lens = tf.contrib.seq2seq.dynamic_decode(decoder=pred_decoder,
                                                                                 maximum_iterations=tf.reduce_max(charseq_lens) + 10)
            lemma_predictions = dec_outputs.predicted_ids[:, :, 0]
            lemma_prediction_lengths = 1 + find_first(lemma_predictions, eow)

    return loss_lem, (lemma_predictions_training, lemma_predictions, lemma_prediction_lengths)


def sense_predictor(word_rnn_outputs, tag_feats, target_senses, num_senses, words_count,
                    predict_sense, sense_smoothing):
    """Network for predicting sense separately"""
    if predict_sense:
        sense_features = word_rnn_outputs
        sense_features = tf.concat([sense_features, tag_feats], axis=-1)
        sense_layer = tf.layers.dense(sense_features, num_senses)
        sense_prediction = tf.argmax(sense_layer, axis=1)
        _gold_senses = tf.one_hot(target_senses, num_senses)
        sense_loss = tf.losses.softmax_cross_entropy(
            _gold_senses, sense_layer, label_smoothing=sense_smoothing)
    else:
        sense_prediction = tf.zeros([words_count], dtype=tf.int64)
        sense_loss = 0.0

    return sense_loss, sense_prediction
