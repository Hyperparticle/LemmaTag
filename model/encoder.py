import tensorflow as tf


def _embed_words(name, word_ids, num_words, we_dim):
    """Create one word embedding"""
    matrix_word_embeddings = tf.get_variable("word_embeddings_{}".format(name),
                                             shape=[num_words, we_dim],
                                             dtype=tf.float32)
    # [sentences, words, dim]
    return tf.nn.embedding_lookup(matrix_word_embeddings, word_ids)


def _embed_characters(name, charseqs, charseq_ids, charseq_lens, word_indexes,
                      num_chars, cle_dim, dropout, is_training):
    """Character-level embeddings"""
    with tf.variable_scope("char_embed_{}".format(name)):
        character_embeddings = tf.get_variable("character_embeddings_{}".format(name),
                                               shape=[num_chars, cle_dim], dtype=tf.float32)
        characters_embedded = tf.nn.embedding_lookup(character_embeddings, charseqs)
        characters_embedded = tf.layers.dropout(characters_embedded, rate=dropout,
                                                training=is_training)

        (output_fwd, output_bwd), (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(cle_dim),
            tf.nn.rnn_cell.GRUCell(cle_dim),
            characters_embedded,
            sequence_length=charseq_lens,
            dtype=tf.float32)

        cle_states = tf.concat([state_fwd, state_bwd], axis=1)
        cle_outputs = tf.concat([output_fwd, output_bwd], axis=1)
        sentence_cle_states = tf.nn.embedding_lookup(cle_states, charseq_ids)
        sentence_cle_outputs = tf.nn.embedding_lookup(cle_outputs, charseq_ids)
        word_cle_states = tf.gather_nd(sentence_cle_states, word_indexes)
        word_cle_outputs = tf.gather_nd(sentence_cle_outputs, word_indexes)

    return sentence_cle_states, word_cle_outputs, word_cle_states


def _sentence_rnn(name, inputs, sentence_lens, rnn_cell, rnn_cell_dim, rnn_layers, dropout, is_training):
    """Sentence-level stacked RNN with residual connections and dropout between layers"""
    hidden_layer = tf.layers.dropout(inputs, rate=dropout, training=is_training)
    for i in range(rnn_layers):
        with tf.variable_scope("word-level-rnn-{}".format(name)):
            (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                rnn_cell(rnn_cell_dim),
                rnn_cell(rnn_cell_dim),
                hidden_layer,
                sequence_length=sentence_lens,
                dtype=tf.float32,
                scope="word-level-rnn-{}-{}".format(name, i))

            hidden_layer += tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=dropout,
                                              training=is_training)
    return hidden_layer


def encoder_network(word_indexes, word_ids, charseqs, charseq_ids, charseq_lens, sentence_lens,
                    num_words, num_chars, we_dim, cle_dim, rnn_cell, rnn_cell_dim, rnn_layers, dropout, is_training,
                    separate_embed, separate_rnn):
    """An encoder consisting of combined character-level and word-level embeddings,
    followed by a bidirectional sentence-level RNN
    """

    # 1. Calculate the word-level embeddings for input to the tag/lemma decoders
    if separate_embed:
        # [sentences, words, dim], [words, char, dim], [words, dim]
        rnn_inputs_lemmas, word_cle_outputs, word_cle_states = _embed_characters("lemmas",
                                                                                 charseqs, charseq_ids, charseq_lens,
                                                                                 word_indexes, num_chars, cle_dim,
                                                                                 dropout, is_training)
        rnn_inputs_lemmas += _embed_words("lemmas", word_ids, num_words, we_dim)

        rnn_inputs_tags_words = _embed_words("tags", word_ids, num_words, we_dim)
        rnn_inputs_tags_characters = _embed_characters("tags",
                                                       charseqs, charseq_ids,
                                                       charseq_lens,
                                                       word_indexes, num_chars,
                                                       cle_dim, dropout,
                                                       is_training)[0]
        rnn_inputs_tags = rnn_inputs_tags_words + rnn_inputs_tags_characters
    else:
        # [sentences, words, dim], [words, char, dim], [words, dim]
        rnn_inputs_lemmas, word_cle_outputs, word_cle_states = _embed_characters("common",
                                                                                 charseqs, charseq_ids, charseq_lens,
                                                                                 word_indexes, num_chars, cle_dim,
                                                                                 dropout, is_training)
        rnn_inputs_lemmas += _embed_words("common", word_ids, num_words, we_dim)
        rnn_inputs_tags = rnn_inputs_lemmas

    # 2. Calculate the sentence-level outputs for input to the tag/lemma decoders
    if separate_rnn:
        sentence_rnn_outputs_tags = _sentence_rnn("tags", rnn_inputs_tags, sentence_lens,
                                                  rnn_cell, rnn_cell_dim, rnn_layers, dropout, is_training)
        _sentence_rnn_outputs_lemma = _sentence_rnn("lemmas", rnn_inputs_lemmas, sentence_lens,
                                                    rnn_cell, rnn_cell_dim, rnn_layers, dropout, is_training)
        word_rnn_outputs = tf.gather_nd(_sentence_rnn_outputs_lemma, word_indexes)
    else:
        sentence_rnn_outputs_tags = _sentence_rnn("common", rnn_inputs_tags, sentence_lens,
                                                  rnn_cell, rnn_cell_dim, rnn_layers, dropout, is_training)
        word_rnn_outputs = tf.gather_nd(sentence_rnn_outputs_tags, word_indexes)

    return rnn_inputs_tags, word_rnn_outputs, sentence_rnn_outputs_tags, word_cle_states, word_cle_outputs
