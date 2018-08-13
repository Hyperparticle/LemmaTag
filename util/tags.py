# Defines classes for processing different types of tags

import numpy as np

class WholeTags:
    """Treats a unique tag string as a distinct tag"""

    def __init__(self, train):
        self._train = train

    def num_tags(self):
        return [(len(self._train.factors[self._train.TAGS].words), 1.0)]

    def accuracy_mask(self):
        return [True]

    def encode(self, tag_ids, seq_ids, seqs):
        return np.expand_dims(tag_ids, -1)

    def decode(self, tags):
        result = []
        for sentence in tags:
            result.append([self._train.factors[self._train.TAGS].words[tag[0]] for tag in sentence])

        return result


class CharTags:
    """Treats each character in a tag string as a tag subcategory"""

    def __init__(self, train, regularization_weight_compositional=1., regularization_weight_whole=1.):
        self._train = train

        self._regularization_weight_compositional = regularization_weight_compositional
        self._regularization_weight_whole = regularization_weight_whole

        self._train_alphabet = train.factors[train.TAGS].alphabet

        self._taglen = len(train.factors[train.TAGS].strings[0][0])

        self._alphabet_maps = [{"<unk>": 0} for _ in range(self._taglen)]
        self._alphabets = [["<unk>"] for _ in range(self._taglen)]
        self._cache = {}
        for tag_id, tag in enumerate(train.factors[train.TAGS].words):
            if len(tag) != self._taglen: continue

            entry = np.zeros(self._taglen + 1, dtype=np.int32)
            entry[0] = tag_id
            for i in range(self._taglen):
                if tag[i] not in self._alphabet_maps[i]:
                    self._alphabet_maps[i][tag[i]] = len(self._alphabets[i])
                    self._alphabets[i].append(tag[i])
                entry[i + 1] = self._alphabet_maps[i][tag[i]]
            self._cache[tag_id] = entry

    def num_tags(self):
        return [(len(self._train.factors[self._train.TAGS].words), self._regularization_weight_whole)] + \
               [(len(alphabet), self._regularization_weight_compositional) for alphabet in self._alphabets]

    def accuracy_mask(self):
        return [True] + [False] * self._taglen

    def encode(self, tag_ids, seq_ids, seqs):
        tags = np.zeros(seq_ids.shape + (self._taglen + 1,), dtype=np.int32)
        for i in range(seq_ids.shape[0]):
            for j in range(seq_ids.shape[1]):
                if tag_ids[i, j] in self._cache:
                    tags[i, j] = self._cache[tag_ids[i, j]]
                else:
                    tags[i, j, 0] = tag_ids[i, j]
                    seq = seqs[seq_ids[i, j]]
                    if len(seq) == self._taglen:
                        for k in range(self._taglen):
                            tags[i, j, k + 1] = self._alphabet_maps[k].get(self._train_alphabet[seq[k] if seq[k] < len(self._train_alphabet) else 0], 0)
                    else:
                        tags[i, j, 1:] = 0

        return tags

    def decode(self, tags):
        result = []
        for sentence in tags:
            result.append([self._train.factors[self._train.TAGS].words[tag[0]] for tag in sentence])

        return result


class DictTags:
    """Treats a tag as composed of character delimited tag subcategories"""

    def __init__(self, train, regularization_weight_compositional=1., regularization_weight_whole=1.):
        self._train = train

        self._regularization_weight_compositional = regularization_weight_compositional
        self._regularization_weight_whole = regularization_weight_whole

        self._train_alphabet = train.factors[train.TAGS].alphabet

        self._taglen = len(train.factors[train.TAGS].strings[0][0])

        self._alphabet_maps = [{"<unk>": 0} for _ in range(self._taglen)]
        self._alphabets = [["<unk>"] for _ in range(self._taglen)]
        self._cache = {}
        for tag_id, tag in enumerate(train.factors[train.TAGS].words):
            if len(tag) != self._taglen: continue

            entry = np.zeros(self._taglen + 1)
            entry[0] = tag_id
            for i in range(self._taglen):
                if tag[i] not in self._alphabet_maps[i]:
                    self._alphabet_maps[i][tag[i]] = len(self._alphabets[i])
                    self._alphabets[i].append(tag[i])
                entry[i + 1] = self._alphabet_maps[i][tag[i]]
            self._cache[tag_id] = entry

    def num_tags(self):
        return [(len(self._train.factors[self._train.TAGS].words), self._regularization_weight_whole)] + \
               [(len(alphabet), self._regularization_weight_compositional) for alphabet in self._alphabets]

    def accuracy_mask(self):
        return [True] + [False] * self._taglen

    def encode(self, tag_ids, seq_ids, seqs):
        tags = np.zeros(seq_ids.shape + (self._taglen + 1,))
        for i in range(seq_ids.shape[0]):
            for j in range(seq_ids.shape[1]):
                if tag_ids[i, j] in self._cache:
                    tags[i, j] = self._cache[tag_ids[i, j]]
                else:
                    tags[i, j, 0] = tag_ids[i, j]
                    seq = seqs[seq_ids[i, j]]
                    if len(seq) == self._taglen:
                        for k in range(self._taglen):
                            tags[i, j, k + 1] = self._alphabet_maps[k].get(self._train_alphabet[seq[k] if seq[k] < len(self._train_alphabet) else 0], 0)
                    else:
                        tags[i, j, 1:] = 0

        return tags

    def decode(self, tags):
        result = []
        for sentence in tags:
            result.append([self._train.factors[self._train.TAGS].words[tag[0]] for tag in sentence])

        return result
