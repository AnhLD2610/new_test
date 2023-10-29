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
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset

from NYTDataset.paths import NYTPaths
from data.dataset_groups import DatasetGroup


def parse_examples(article_file, summary_file, examples):
    with open(article_file, 'rb') as article:
        with open(summary_file, 'rb') as summary:
            article_lines = article.readlines()
            summary_lines = summary.readlines()
            for i in range(len(article_lines)):
                examples.append((article_lines[i], summary_lines[i]))


class NYTDataset(Dataset):
    def __init__(self, dataset, group=DatasetGroup.train, num_chunks=None, num_examples=None):
        super(Dataset, self).__init__()
        if type(dataset) is str:
            root_dir = dataset
            self.nyt = NYTPaths(root_dir)
        elif type(dataset) is NYTPaths:
            self.nyt = dataset

        self.examples = list()
        self._loaded = False
        self.group = group
        self.num_examples = num_examples

    def get_pairs(self):
        if self.group == DatasetGroup.train:
            return self.nyt.train_pairs
        elif self.group == DatasetGroup.validation:
            return self.nyt.valid_pairs
        else:
            return self.nyt.test_pairs

    def _load(self):
        if self._loaded: return
        pairs = self.get_pairs()

        num_threads = min(len(pairs), os.cpu_count() - 1) or 1
        pool = ThreadPool(num_threads)
        args = [(article, summary, self.examples) for article, summary in pairs]

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
