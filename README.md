# LemmaTag

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

The following project provides source code for the paper, ["LemmaTag: Jointly Tagging and Lemmatizing for Morphologically-Rich Languages with BRNNs"](https://www.researchgate.net/publication/326960698_LemmaTag_Jointly_Tagging_and_Lemmatizing_for_Morphologically-Rich_Languages_with_BRNNs).

## Overview

The image below provides a detailed overview of the architecture and design of the system.

<img src="images/model.png" alt="Model"> <img src="images/tag-components.png" alt="Tag Components" width="400">


- **Top** - Lemma decoder, consisting of a standard seq2seq autoregressive decoder with Luong attention on character encodings, and with additional inputs of processed tagger features `T_i`, embeddings `e^w_i` and sentence-level outputs `o^w_i`. Gradients from the lemmatizer are stopped from flowing into the tagger (denoted `GradStop`).
- **Middle** - Sentence-level encoder and tag classifier. Two BRNN layers with residual connections act on the embedded words `e^w_i` of a sentence, providing context. The output of the tag classification are the logits for both the whole tags `t_i` and their components `t_{i,j}`.
- **Bottom Left** - Word-level encoder. The characters of every input word are embedded with a look-up table and encoded with a BRNN. The outputs are used in decoder attention, the final states are summed with the word-level embedding. `WD` denotes word-dropout.

- **Bottom Right** - The tag components of the PDT Czech treebank with the numbers of valid values. Around 1500 different tags are in use in the PDT.

Thick slanted lines denote training dropout.

## Getting Started

### Requirements

- Python 3.5+
- Tensorflow 1.6+

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
