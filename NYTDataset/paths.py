#!/usr/bin/env python
"""
File: nyt
Date: 2/10/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os


class NYTPaths(object):
    """ This class is meant simply to provide an abstraction
    around the layout of the data set on the filesystem and nothing else!
    The idea being that you just put the data in the specified layout
    and provide a single path to it and then never worry about file locations again.
    """

    def __init__(self, path, small=False):
        self.path = path
        self.small = small

    @property
    def vocab_path(self):
        return os.path.join(self.preprocessed_path, "vocab.txt")

    @property
    def preprocessed_path(self):
        preprocessed = "preprocessed-small" if self.small else "preprocessed"
        return os.path.join(self.data_path, preprocessed)

    @property
    def data_path(self):
        return os.path.join(self.path, "data")

    @property
    def vocab(self):
        with open(self.vocab_path, 'r') as f:
            lines = f.readlines()
            split_lines = map(lambda line: tuple(line.split()), lines)
        vocab = {}
        for entry in split_lines:
            if len(entry) == 2:
                word, index = entry
                vocab[word] = int(index)
        return vocab
        # return {word: int(index) for word, index in split_lines}

    @property
    def train_path(self):
        return os.path.join(self.preprocessed_path, "train")

    @property
    def train_files(self):
        return files_in(self.train_path)

    @property
    def train_article_files(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("article"), self.train_files))

    @property
    def train_summary_files(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("summary"), self.train_files))

    @property
    def train_pairs(self):
        return self.file_pairs(self.train_article_files)

    @property
    def valid_path(self):
        return os.path.join(self.preprocessed_path, "validation")

    @property
    def valid_files(self):
        return files_in(self.valid_path)

    @property
    def valid_article_files(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("article"), self.valid_files))

    @property
    def valid_pairs(self):
        return self.file_pairs(self.valid_article_files)

    @property
    def test_path(self):
        return os.path.join(self.preprocessed_path, "test")

    @property
    def test_files(self):
        return files_in(self.test_path)

    @property
    def test_article_files(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("article"), self.test_files))

    @property
    def test_pairs(self):
        return self.file_pairs(self.test_article_files)

    def file_pairs(self, article_files):
        pairs = []
        for article in article_files:
            summary = article.replace("article", "summary")
            pairs.append((article, summary))
        return pairs

def files_in(dir):
    files = list()
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files
