import contextlib
import time
import sys
import numpy as np
from logging import warning, info, debug, error


class Tee:
    """
    Redirect stdin+stdout to a file and at the same time to the orig stdin+stdout.
    Use as a context manager or with .start() and .stop().
    """

    def __init__(self, name, mode="wt"):
        self.filename = name
        self.mode = mode

    def __enter__(self):
        self.start()

    def __exit__(self, *exceptinfo):
        self.stop()

    def start(self):
        self.file = open(self.filename, self.mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def stop(self, *exceptinfo):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()



@contextlib.contextmanager
def log_time(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time.time()
    try:
        yield
    finally:
        dt = time.time() - start
        if dt < 300:
            info('{} took {:.2f} s'.format(prefix, dt))
        else:
            info('{} took {:.2f} s ({:.2f} min)'.format(prefix, dt, dt / 60.0))


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}
        tag_sets = [set() for _i in range(15)]
        self.tags = set()
        self.maxlen = 0

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                    for iv, v in enumerate(columns[i + 1]):
                        tag_sets[iv].add(v)
                    self.tags.add(columns[i + 1])
                self.maxlen = max(self.maxlen, len(analyses))
                self.analyses[columns[0]] = analyses
        self.tags = ["<unk>"] + list(self.tags)
        # self.tagchar_values = [list(vset) for vset in tag_sets]
        # self.tagchar_dicts = [{v: i for i, v in enumerate(vlist)} for vlist in self.tagchar_values]
        # self.tagchar_count = [len(vlist) for vlist in self.tagchar_values]
        self.tag_dict = {v: i for i, v in enumerate(self.tags)}

    def get(self, word):
        return self.analyses.get(word, [])

    def get_tags(self, word):
        return [lt.tag for lt in self.get(word)]

    def get_tag_ids(self, word):
        return [self.tag_dict.get(lt.tag, 0) for lt in self.get(word)]

    def get_tag_ids_len_array(self, word, length):
        ids = self.get_tag_ids(word)[:length]
        return (len(ids), np.array(ids + [0] * (length - len(ids))))
