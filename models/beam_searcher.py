#!/usr/bin/env python
"""
File: beam_searcher
Date: 3/9/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Refactored and slightly modified from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Translator.py
"""

import torch

from data.batch import Batch
from transformer.Beam import Beam


class BeamSearcher:

    def __init__(self, model, beam_size, num_best=1, ngram_block_size=None, device=False):
        self.model = model
        self.beam_size = beam_size
        self.device = device
        self.ngram_block_size = ngram_block_size
        self.num_best = num_best

    def search_batch(self, src: Batch, src_enc, max_token_seq_len):
        """ Performs beam search on a batch of source examples """

        src_seq = src.sequence
        extra = []

        # Repeat data for beam search
        num_instances, src_len, d_h = src_enc.size()
        src_seq = src_seq.repeat(1, self.beam_size).view(num_instances * self.beam_size, src_len)
        src_enc = src_enc.repeat(1, self.beam_size, 1).view(num_instances * self.beam_size, src_len, d_h)

        # Prepare beams
        inst_dec_beams = [Beam(self.beam_size, device=self.device) for _ in range(num_instances)]

        # Bookkeeping for active or not
        active_inst_idx_list = list(range(num_instances))
        inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # Decode
        for len_dec_seq in range(1, max_token_seq_len + 1):
            active_inst_idx_list, extra_info = self.beam_decode_step(inst_dec_beams, len_dec_seq, src, src_seq, src_enc,
                                                         inst_idx_to_position_map)
            extra.append(extra_info)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>

            src_seq, src_enc, inst_idx_to_position_map = self.collate_active_info(src_seq, src_enc, inst_idx_to_position_map,
                                                                                  active_inst_idx_list)

        batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams)
        final_extra = self.model.decode_extra_info(src_seq, batch_hyp, extra)

        return batch_hyp, batch_scores, final_extra

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, src, src_seq, enc_output, inst_idx_to_position_map):
        """ Decode and update beam status, and then return active beam idx """
        n_active_inst = len(inst_idx_to_position_map)

        dec_seq = self.prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = self.prepare_beam_dec_pos(len_dec_seq, n_active_inst)
        
        active_idxs = torch.LongTensor([i for i, b in enumerate(inst_dec_beams) if not b.done]).to(self.device)

        p_vocab, extra_info = self.model.decode_one_step(dec_seq, dec_pos, src, src_seq, enc_output, active_idxs)
        p_vocab = p_vocab.view(n_active_inst, self.beam_size, -1)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = self.collect_active_inst_idx_list(inst_dec_beams, p_vocab, inst_idx_to_position_map)

        return active_inst_idx_list, extra_info

    def collate_active_info(self,  src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

        active_src_seq = self.collect_active_part(src_seq, active_inst_idx, n_prev_active_inst)
        active_src_enc = self.collect_active_part(src_enc, active_inst_idx, n_prev_active_inst)
        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def collect_hypothesis_and_scores(self, inst_dec_beams):
        all_hypotheses = list()
        all_scores = list()
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()

            all_scores.append(scores[0])

            best_idx = tail_idxs[0]  # we only want the best hypothesis

            hyp = inst_dec_beams[inst_idx].get_hypothesis(best_idx, without_stop_id=True)
            all_hypotheses.append(hyp)

        return all_hypotheses, all_scores

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst):
        """ Collect tensor parts associated to active instances. """

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * self.beam_size, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def prepare_beam_dec_seq(self, inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def prepare_beam_dec_pos(self, len_dec_seq, n_active_inst):
        dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
        dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * self.beam_size, 1)
        return dec_partial_pos

    def collect_active_inst_idx_list(self, inst_beams, word_prob, inst_idx_to_position_map):
        active_instance_indices = list()
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], ngram_block_size=self.ngram_block_size)
            if not is_inst_complete:
                active_instance_indices.append(inst_idx)

        return active_instance_indices

    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        """ Indicate the position of an instance in a tensor. """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
