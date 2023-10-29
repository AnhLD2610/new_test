#!/usr/bin/env python
"""
File: dataset
Date: 2/11/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os
import struct
import re
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset
from tensorflow.core.example import example_pb2

from CNNDailyMail import CNNDailyMailPaths
from data.dataset_groups import DatasetGroup
import data
from utils import cast_to_string

def filter_start_end_tokens(txt):
    txt_str = cast_to_string(txt)
    txt_str = re.sub(data.SENTENCE_START, "", txt_str)
    txt_str = re.sub(data.SENTENCE_END, "", txt_str)
    return txt_str.encode()

def get_parts(example):
    feature = example.features.feature
    article = feature['article'].bytes_list.value[0]
    abstract = feature['abstract'].bytes_list.value[0]
    abstract_clean = filter_start_end_tokens(abstract)
    return article, abstract_clean


def parse_examples(file, examples):
    with open(file, 'rb') as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes: break  # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, f.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            article, abstract = get_parts(e)
            if len(article) == 0 or len(abstract) == 0:
                continue
            examples.append((article, abstract))


class CNNDMDataset(Dataset):
    def __init__(self, dataset, group=DatasetGroup.train, num_chunks=None, num_examples=None):
        super(Dataset, self).__init__()
        if type(dataset) is str:
            root_dir = dataset
            self.daily_mail = CNNDailyMailPaths(root_dir)
        elif type(dataset) is CNNDailyMailPaths:
            self.daily_mail = dataset

        self.num_chunks = num_chunks
        self.examples = list()
        self._loaded = False
        self.group = group
        self.num_examples = num_examples

    def get_chunks(self):
        if self.group == DatasetGroup.train:
            return self.daily_mail.train_chunks
        elif self.group == DatasetGroup.validation:
            return self.daily_mail.valid_chunks
        else:
            return self.daily_mail.test_chunks

    def _load(self):
        if self._loaded: return
        chunks = self.get_chunks()
        if self.num_chunks is not None:
            chunks = chunks[:self.num_chunks]

        num_threads = min(len(chunks), os.cpu_count() - 1) or 1
        pool = ThreadPool(num_threads)
        args = [(chunk_file, self.examples) for chunk_file in chunks]

        print("Loading examples...")
        pool.starmap(parse_examples, args)
        print("Loading complete.")

        if self.num_examples is not None:
            self.examples = self.examples[:self.num_examples]

        self._loaded = True


    @property
    def max_sequence_length_article(self):
        if not self._loaded: self._load()
        return max(map(lambda ex: len(ex[0].split()), self.examples))

    @property
    def max_sequence_length_summary(self):
        if not self._loaded: self._load()
        return max(map(lambda ex: len(ex[1].split()), self.examples))

    def __len__(self):
        if not self._loaded: self._load()
        return len(self.examples)

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        return self.examples[index]
