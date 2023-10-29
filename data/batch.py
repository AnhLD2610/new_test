#!/usr/bin/env python
"""
File: batch
Date: 3/7/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Based on:
https://github.com/abisee/pointer-generator/blob/master/batcher.py
"""


import torch
import torch.autograd.variable as Variable
from typing import List
import numpy as np

import data
from data.vocab import article2ids, abstract2ids, pad_sents

class Batch:

    def __init__(self, batch, max_seq_len, vocab, device=torch.device('cpu')):

        self.batch = batch
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.device = device

        self.summary = None
        self._sequence = None
        self._positions = None

        #oov data
        self.ids_with_oov_ids = None
        self.oovs = None
        self.extra_zeros = None
        self._loaded = False

    def _load(self):
        if self._loaded: return
        self._sequence, self._positions = self.vocab.to_input_tensor(self.batch, device=self.device,
                                                                  seq_len=self.max_seq_len,
                                                                  with_position=True)
        self._loaded = True

    def get_seq_and_pos(self):
        return self.sequence, self.positions

    @property
    def sequence(self):
        if not self._loaded:
            self._load()
        return self._sequence

    @property
    def positions(self):
        if not self._loaded:
            self._load()
        return self._positions

    def get_oov_data_src(self, repeat=None):
        if self.ids_with_oov_ids is None or self.oovs is None or self.extra_zeros is None:
            self.ids_with_oov_ids, self.oovs, self.extra_zeros = self.to_article_oov()

        ids = self.ids_with_oov_ids
        extra_zeros = self.extra_zeros
        if repeat is not None:
            batch_size, src_len = ids.shape
            ids = ids.repeat(1, repeat).view(-1, src_len)
            if extra_zeros is not None:
                batch_size, n_extra = extra_zeros.shape
                extra_zeros = extra_zeros.repeat(1, repeat).view(-1, n_extra)

        return ids, self.oovs, extra_zeros

    def get_oovs_src(self):
        if self.oovs is None:
            self.get_oov_data_src()
        return self.oovs

    def get_oov_ids_tgt(self, article_oovs):
        if self.ids_with_oov_ids is None:
            self.ids_with_oov_ids = self.to_abstract_oov(article_oovs)
        return self.ids_with_oov_ids


    #######

    def to_article_oov(self):
        tokenized = [sent.split() for sent in self.batch]
        padded_sents = pad_sents(tokenized, data.PAD_TOKEN, length=self.max_seq_len, with_position=False)
        batch_with_oov_ids = np.zeros((len(self.batch), self.max_seq_len))
        batch_oovs = []
        for i, sent in enumerate(padded_sents):
            ids_with_oovs, oovs = article2ids(sent, self.vocab)
            batch_with_oov_ids[i, :] = ids_with_oovs[:]
            batch_oovs.append(oovs)
        max_oovs = max([len(batch) for batch in batch_oovs])
        extra_zeros = None
        if max_oovs > 0:
            extra_zeros = torch.zeros((len(self.batch), max_oovs), dtype=torch.float, device=self.device)
        batch_with_oov_ids = torch.from_numpy(batch_with_oov_ids).long().to(self.device)
        return batch_with_oov_ids, batch_oovs, extra_zeros

    def to_abstract_oov(self, article_oovs: List[str]):
        tokenized = [sent.split() for sent in self.batch]
        padded_sents = pad_sents(tokenized, data.PAD_TOKEN, length=self.max_seq_len, with_position=False)
        batch_with_oov_ids = [abstract2ids(sent, self.vocab, article_oovs[i]) for i,sent in enumerate(padded_sents)]
        return torch.LongTensor(batch_with_oov_ids).to(self.device)

