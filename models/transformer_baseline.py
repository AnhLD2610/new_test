#!/usr/bin/env python
"""
File: transformer
Date: 2/15/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Model saving/loading adapted from:
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import HyperParameters
from transformer.Models import Transformer


class Baseline(nn.Module):

    def __init__(self, hyperams: HyperParameters, vocab, word_embeddings=None, device=torch.device('cpu'), compute_extra=False):
        super(Baseline, self).__init__()

        self.word_embeddings = word_embeddings
        self.hyperams = hyperams
        self.vocab = vocab
        self.device = device
        self.compute_extra = compute_extra

        self.transformer = Transformer(
            len(vocab), len(vocab),
            hyperams.max_seq_len_src, hyperams.max_seq_len_tgt,
            d_word_vec=hyperams.d_model,
            d_model=hyperams.d_model,
            d_inner=hyperams.d_inner,
            n_layers=hyperams.transformer_layers,
            n_head=hyperams.n_head,
            d_k=hyperams.d_k,
            d_v=hyperams.d_v,
            dropout=hyperams.dropout,
            tgt_emb_prj_weight_sharing=hyperams.tgt_emb_prj_weight_sharing,
            emb_src_tgt_weight_sharing=hyperams.emb_src_tgt_weight_sharing,
            word_embeddings=word_embeddings,
            freeze_embeddings=hyperams.freeze_embeddings)

    def get_state(self):
        state = {
            'encoder_state_dict': self.transformer.encoder.state_dict(),
            'decoder_state_dict': self.transformer.decoder.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        epoch = state_dict['epoch']
        optimizer_state = state_dict['optimizer']
        current_loss = state_dict['current_loss']
        encoder_state = state_dict['encoder_state_dict']
        decoder_state = state_dict['decoder_state_dict']

        self.transformer.encoder.load_state_dict(encoder_state)
        self.transformer.decoder.load_state_dict(decoder_state)

    def forward(self, src, tgt, log_dict=False):
        result = F.log_softmax(self.transformer(src, tgt))
        if log_dict:
            return result, {}
        else:
            return result

    def decode_one_step(self, dec_seq, dec_pos, src, src_seq, enc_output, active_idxs):
        vocab_logits, dec_output, enc_dec_attns = self.transformer.decode_one_step(dec_seq, dec_pos, src_seq, enc_output)
        p_vocab = F.log_softmax(vocab_logits, dim=1)
        extra_info = None
        return p_vocab, extra_info

    def decode_extra_info(self, src_seq, batch_hyp, extra):
        return None

    def gold(self, src, tgt):
        target_batch, target_pos = tgt.get_seq_and_pos()
        return target_batch[:, 1:]
