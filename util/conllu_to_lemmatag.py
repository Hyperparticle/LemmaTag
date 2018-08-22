#!/usr/bin/env python3

# Converts CoNLL format of Universal Dependency (UD) files to LemmaTag format
# See http://universaldependencies.org/format.html

column_names = [
    "ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"
]

column_pos = {name: i for i, name in enumerate(column_names)}


def conllu_to_lemmatag(lines, pos_column="XPOS", max_lines=None):
    line_count = 0

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        elif line == "":
            line_count = 0
            yield ""
        else:
            if max_lines and line_count and line_count >= max_lines:
                continue

            line_count += 1
            tokens = line.split("\t")
            yield "\t".join([tokens[column_pos["FORM"]], tokens[column_pos["LEMMA"]], tokens[column_pos[pos_column]]])


if __name__ == "__main__":
    import sys

    for lemma_tag in conllu_to_lemmatag(sys.stdin):
        print(lemma_tag)
