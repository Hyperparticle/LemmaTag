# LemmaTag

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![TensorFlow 1.10](https://img.shields.io/badge/TensorFlow-1.10-orange.svg)](https://www.tensorflow.org/install/) [![Python 3.5+](https://img.shields.io/badge/Python-3.5+-yellow.svg)](https://www.python.org/downloads/)

The following project provides a neural network architecture for [part-of-speech tagging](https://medium.com/greyatom/learning-pos-tagging-chunking-in-nlp-85f7f811a8cb) and [lemmatizing](https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/) sentences, achieving state-of-the-art results in morphologically-rich languages: Czech, German, and Arabic [(Kondratyuk et al., 2018)](https://www.researchgate.net/publication/326960698_LemmaTag_Jointly_Tagging_and_Lemmatizing_for_Morphologically-Rich_Languages_with_BRNNs).

## Overview

There are two main ideas to LemmaTag:

1. Sharing the initial layers of the network is mutually beneficial for part-of-speech tagging and lemmatization, as they are similar tasks. This results in higher accuracy and requires less training time.
2. The lemmatizer can further improve its accuracy by looking at the tagger's predictions, i.e., taking the output of the tagger as an additional lemmatizer input.

### Model

The model consists of 3 parts:

- The **shared encoder** generates [character-level](http://colinmorris.github.io/blog/1b-words-char-embeddings) and [word-level embeddings](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/) and processes them through a [bidirectional RNN (BRNN)](https://towardsdatascience.com/introduction-to-sequence-models-rnn-bidirectional-rnn-lstm-gru-73927ec9df15).
- The **tagger decoder** generates part-of-speech tags with a [softmax classifier](https://becominghuman.ai/making-a-simple-neural-network-classification-2449da88c77e) by using the output of the shared encoder.
- The **lemmatizer decoder** generates lemmas character-by-character with an RNN by using the outputs of the shared encoder and also the output of the tagger.

The image below provides a detailed overview of the architecture and design of the system. For technical details, see the paper, ["LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs"](https://www.researchgate.net/publication/326960698_LemmaTag_Jointly_Tagging_and_Lemmatizing_for_Morphologically-Rich_Languages_with_BRNNs).

![Model](images/model.png)

- **Bottom** - Word-level encoder, with word input `w`, character inputs `c`, character states `e^c`, and combined word embedding `e^w`. Thick slanted lines denote [training dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5).
- **Top Left** - Sentence-level encoder and tag classifier, with word-level inputs `e^w`. Two BRNN layers with residual connections act on the embedded words of a sentence, producing intermediate sentence contexts `o^w` and tag classification `t`.
- **Top Right** - Lemma decoder, consisting of a [seq2seq decoder](https://medium.com/@devnag/seq2seq-the-clown-car-of-deep-learning-f88e1204dac3) with [attention](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/) on character encodings `e^c`, and with additional inputs of processed tagger features `t`, embeddings `e^w` and sentence-level contexts `o^w`.

### Morphology Tagging

Not all languages are alike when part-of-speech tagging. For instance, the Czech language has over 1500 different types of tags, while English has about 50. This discrepancy is due to Czech being a morphologically-rich language, which alters the ending of a word to modify aspects like case, number, and gender. English, on the other hand, gets around this by relying heavily on the positioning of a word relative to other words.

The image below shows how Czech tags are split up into several subcategories that delineate a word's [morphology](http://all-about-linguistics.group.shef.ac.uk/branches-of-linguistics/morphology/what-is-morphology/), along with the number of unique values in each subcategory.

![Tag Components](images/tag-components-small.png)

LemmaTag takes advantage of this by also predicting each tag subcategory and feeding this information to the lemmatizer (if the subcategories exist for the language). This modification improves both tagging and lemmatizing accuracies.

## Getting Started

### Requirements

The code uses Python 3.5+ running TensorFlow (tested working with TF 1.10).

1. Clone the repository.

```bash
git clone https://github.com/Hyperparticle/LemmaTag.git
cd ./LemmaTag
```

2. Install the python packages in requirements.txt if you don't have them already.

```bash
pip install -r ./requirements.txt
```

### Downloading the Dataset

### Training and Testing

To start training with default parameters, run

```bash
python lemmatag.py
```

This will save the model periodically and output the training/validation accuracy.

After training and saving the model to a checkpoint file, one may evaluate using

```bash
python lemmatag.py --only_eval
```

Run `python lemmatag.py --help` for a list of all supported arguments.
