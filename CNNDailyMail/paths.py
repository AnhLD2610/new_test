#!/usr/bin/env python
"""
File: CNNDailyMail
Date: 2/10/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os
from cached_property import cached_property


class CNNDailyMailPaths(object):
    """ This class is meant simply to provide an abstraction
    around the layout of the data set on the filesystem and nothing else!
    The idea being that you just put the data in the specified layout
    and provide a single path to it and then never worry about file locations again.
    """

    def __init__(self, path):
        self.path = path

    @property
    def vocab_path(self):
        return os.path.join(self.finished_files, "vocab")

    @property
    def preprocessed_path(self):
        return os.path.join(self.path, "preprocessed")

    @property
    def vocab(self):
        with open(self.vocab_path, 'r') as f:
            lines = f.readlines()
            split_lines = map(lambda line: tuple(line.split()), lines)
        return {word: int(index) for word, index in split_lines}

    @property
    def finished_files(self):
        return os.path.join(self.preprocessed_path, "finished_files")

    @property
    def train_path(self):
        return os.path.join(self.finished_files, "train.bin")

    @property
    def val_path(self):
        return os.path.join(self.finished_files, "val.bin")

    @property
    def test_path(self):
        return os.path.join(self.finished_files, "test.bin")

    @property
    def chunked_path(self):
        return os.path.join(self.finished_files, "chunked")

    @cached_property
    def chunks(self):
        return files_in(self.chunked_path)

    @property
    def train_chunks(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("train"), self.chunks))

    @property
    def valid_chunks(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("val"), self.chunks))

    @property
    def test_chunks(self):
        return list(filter(lambda fn: os.path.basename(fn).startswith("test"), self.chunks))

    def chunk(self, i):
        fn = "train_%03d.bin" % i
        return os.path.join(self.chunked_path, fn)


def files_in(dir):
    files = list()
    for file in os.listdir(dir):
        full_path = os.path.join(dir, file)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files

