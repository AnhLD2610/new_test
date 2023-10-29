#!/usr/bin/env python
"""
File: train
Date: 3/9/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Heavily adapted from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from data.batch import Batch
from models.beam_searcher import BeamSearcher


class TransformerSummarizer(object):  # analogous to "Translator"

    def __init__(self,
                 model,
                 vocab,
                 max_token_seq_len,
                 beam_size=4,
                 num_best=5,
                 n_gram_block_size=None,
                 device=torch.device('cpu')):

        self.device = device
        self.max_token_seq_len = max_token_seq_len
        self.beam_size = beam_size
        self.num_best = num_best
        self.vocab = vocab
        self.ngram_block_size = n_gram_block_size

        self.model = model.to(self.device)

        self.transformer = model.transformer
        self.transformer.word_prob_prj = nn.LogSoftmax(dim=1)

        self.beam_searcher = BeamSearcher(model, self.beam_size, num_best=num_best,
                                          ngram_block_size=self.ngram_block_size, device=device)

    def set_ngram_block_size(self, new_size):
        self.ngram_block_size = new_size
        self.beam_searcher.ngram_block_size = new_size

    def batch_ids_to_words(self, src, batch_hyp):
        _, oovs, _ = src.get_oov_data_src()
        batch_hyp = [self.vocab.indices2words(hyp, oovs=oovs[idx]) for idx, hyp in enumerate(batch_hyp)]
        batch_hyp = [" ".join(dec_out) for dec_out in batch_hyp]
        return batch_hyp

    def summarize_batch(self, src: Batch):
        """ Computes the full summarization prediction, given
        a batch of source sequences """

        self.src = src
        src_seq, src_pos = src.get_seq_and_pos()
        extra_info = None

        with torch.no_grad():
            # Encode
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            src_enc, *_ = self.transformer.encoder(src_seq, src_pos)

            hyps, scores, extra_info = self.beam_searcher.search_batch(src, src_enc, self.max_token_seq_len)

        hyps = self.batch_ids_to_words(src, hyps)
        return hyps, scores, extra_info

