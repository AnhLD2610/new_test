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

from models import HyperParameters
from models.coverage import Coverage
from transformer.Models import Transformer
from models.pointer_generator import PointerAttention, PointerGeneratorNetwork
import data
import copy
import logging
logger=logging.getLogger()


class TransformerPointer(nn.Module):

    def __init__(self, hyperams: HyperParameters, vocab, word_embeddings=None, device=torch.device('cpu'), compute_extra=False):
        super(TransformerPointer, self).__init__()

        self.word_embeddings = word_embeddings
        self.hyperams = hyperams
        self.vocab = vocab
        self.device = device
        self.coverage_loss = None
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

        emb_dim = word_embeddings.shape[1] if word_embeddings is not None else hyperams.d_model
        self.pointer_generator_network = PointerGeneratorNetwork(hidden_size=emb_dim, context_size=emb_dim,
                                                                 device=device)
        self.pointer_attention = PointerAttention(hyperams.n_head, hyperams.max_seq_len_src, device=device)

    def get_state(self):
        state = {
            'encoder_state_dict': self.transformer.encoder.state_dict(),
            'decoder_state_dict': self.transformer.decoder.state_dict(),
            'pointer_state_dict': self.pointer_generator_network.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        epoch = state_dict['epoch']
        optimizer_state = state_dict['optimizer']
        current_loss = state_dict['current_loss']
        encoder_state = state_dict['encoder_state_dict']
        decoder_state = state_dict['decoder_state_dict']
        pointer_state = state_dict.get('pointer_state_dict')

        self.transformer.encoder.load_state_dict(encoder_state)
        self.transformer.decoder.load_state_dict(decoder_state)
        if pointer_state:
            self.pointer_generator_network.load_state_dict(pointer_state)
        else:
            logger.warning("Restoring Pointer Generator with out a state.")

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def filter_src_oovs(self, src, active_idxs):
        batch_with_oov_ids, oovs, extra_zeros = src.get_oov_data_src()
        batch_size, src_len = batch_with_oov_ids.shape
        batch_size, n_extra = extra_zeros.shape

        batch_with_oov_ids = torch.index_select(batch_with_oov_ids, 0, active_idxs)
        extra_zeros = torch.index_select(extra_zeros, 0, active_idxs)

        # todo: remove beam size from this class
        batch_with_oov_ids = batch_with_oov_ids.repeat(1, self.hyperams.beam_size).view(-1, src_len)
        extra_zeros = extra_zeros.repeat(1, self.hyperams.beam_size).view(-1, n_extra)

        return batch_with_oov_ids, oovs, extra_zeros

    def forward(self, src, tgt, log_dict=False):

        enc_output, dec_output, enc_dec_attns = self.transformer.encoder_decoder_hidden(src, tgt)
        vocab_logits = self.transformer.make_vocab_logits(dec_output)

        src_seq, _ = src.get_seq_and_pos()
        tgt_seq, _ = tgt.get_seq_and_pos()
        tgt_seq = tgt_seq[:, :-1]
        _, src_len = src_seq.shape
        _, tgt_len = tgt_seq.shape

        src_oov_data = src.get_oov_data_src(repeat=tgt_len)
        attn_dist, c_t = self.pointer_attention(enc_dec_attns, enc_output, src_seq)

        attn_dist_reshaped = attn_dist.contiguous().view(-1, tgt_len, src_len)
        coverage_vecs = torch.cumsum(attn_dist_reshaped[:, :tgt_len - 1, :], 1)
        attn_vecs = attn_dist_reshaped[:, 1:, :]
        min_vecs = torch.min(coverage_vecs, attn_vecs)
        self.coverage_loss = torch.sum(min_vecs).item()

        # input = tgt_seq, hidden = dec_output, context=c_t
        final_dist, _, pgen = self.pointer_generator_network(tgt_seq, c_t, dec_output, vocab_logits, attn_dist, src_oov_data)

        if log_dict:
            log_dict = { "Avg_Pgen" : torch.mean(pgen).item() }
            if self.hyperams.is_coverage:
                log_dict = { "Avg_Pgen" : torch.mean(pgen).item(), "Coverage Loss" : self.coverage_loss}
            return final_dist, log_dict

        else:
            return final_dist

    def decode_one_step(self, dec_seq, dec_pos, src, src_seq, enc_output, active_idxs):

        # replace oov ids in the decoder sequence with <UNK>
        dec_seq_without_oov = torch.where((dec_seq >= len(self.vocab)) | (dec_seq < 0),
                                          torch.tensor(data.UNK_ID, device=self.device), dec_seq)

        vocab_logits, dec_output, enc_dec_attns = self.transformer.decode_one_step(dec_seq_without_oov, dec_pos, src_seq, enc_output)

        dec_oov = self.filter_src_oovs(src, active_idxs)

        # select only ones with active indexes
        attn_dist, c_t = self.pointer_attention(enc_dec_attns, enc_output, src_seq, decode=True)
        final_dist, attn_dist_, _ = self.pointer_generator_network(dec_seq, c_t, dec_output, vocab_logits, attn_dist,
                                                                   dec_oov,
                                                                   decode=True)
        extra_info = attn_dist

        return final_dist, extra_info

    def decode_extra_info(self, src_seq, batch_hyp, attn_dists):
        if not self.compute_extra:
            return None

        def get_score(ex_idx, dec_idx, dec_id):
            src_i = src_seq[ex_idx].tolist()
            attn_i = attn_dists[dec_idx][ex_idx].tolist()
            if dec_id in src_i:
                src_idx = src_i.index(dec_id)
                return attn_i[src_idx]
            return -1

        pointed_scores = copy.deepcopy(batch_hyp)#[[0]*len(batch_hyp[0])]*len(batch_hyp)
        for i,ex in enumerate(batch_hyp):
            for j,word_id in enumerate(ex):
                pointed_scores[i][j] = get_score(i, j, word_id)

        return pointed_scores


    def gold(self, src, tgt):
        tgt_with_oov_ids = tgt.get_oov_ids_tgt(article_oovs=src.get_oovs_src())
        return tgt_with_oov_ids[:, 1:]
