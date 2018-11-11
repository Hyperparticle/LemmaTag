import tensorflow as tf


def tag_decoder(gold_tags, sentence_rnn_outputs, sentence_weights,
                sentence_sum_weights, num_tags, tag_type, label_smoothing):
    """Applies a tagger decoder network over the sentence-level RNN. Returns tag predictions, loss and accuracy."""
    predictions, loss_tag = [], 0
    tag_outputs = []
    for i, (tags, weight) in enumerate(num_tags):
        output_layer = tf.layers.dense(sentence_rnn_outputs, tags)
        tag_outputs.append(output_layer)
        predictions.append(tf.argmax(output_layer, axis=2, output_type=tf.int32))

        # Training
        if label_smoothing:
            gold_labels = tf.one_hot(gold_tags[:, :, i], tags) * (1 - label_smoothing) + label_smoothing / tags
            loss_tag += tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=sentence_weights) * weight
        else:
            loss_tag += tf.losses.sparse_softmax_cross_entropy(gold_tags[:, :, i], output_layer, weights=sentence_weights) * weight
    tag_outputs = tf.concat(tag_outputs, axis=-1)  # Tagger output features for lemmatizer
    predictions = tf.stack(predictions, axis=-1)

    correct_tag = tf.reduce_sum(tf.cast(tf.reduce_all(
        tf.logical_or(tf.equal(gold_tags, predictions), tf.logical_not(tag_type.accuracy_mask())),
        axis=2), tf.float32) * sentence_weights) / sentence_sum_weights
    correct_tags_compositional = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(  # Average accuracy of all tags
        gold_tags, predictions), axis=2), tf.float32) * sentence_weights) / sentence_sum_weights

    return loss_tag, tag_outputs, predictions, correct_tag, correct_tags_compositional


def tag_features(tag_outputs, word_indexes, words_count, rnn_cell_dim, dropout, is_training,
                 no_tags_to_lemmas, tag_signal_dropout):
    """Tagger output features for the lemmatizer"""
    if no_tags_to_lemmas:
        tag_feats = tf.zeros([words_count, rnn_cell_dim], dtype=tf.float32)
    else:
        tag_feats = tag_outputs
        tag_feats = tf.stop_gradient(tag_feats)
        tag_feats = tf.layers.dense(tag_feats, rnn_cell_dim, activation=tf.nn.relu)
        tag_feats = tf.layers.dropout(tag_feats, rate=dropout, training=is_training)
        tag_feats = tf.gather_nd(tag_feats, word_indexes)

    # Renormed word-level signal dropout
    if tag_signal_dropout:
        dropout_norm = 1 - tf.cast(is_training, tf.float32) * tag_signal_dropout
        tag_feats = tf.layers.dropout(tag_feats, noise_shape=[words_count, 1],
                                      rate=tag_signal_dropout,
                                      training=is_training) * dropout_norm
    
    return tag_feats
