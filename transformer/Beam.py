#!/usr/bin/env python
"""
File: transformer
Date: 2/15/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Modified from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Beam.py

Which was in turn heavily borrowed from:
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py

Our modifications:
- Added ngram blocking (lines 63-64, 121-133)
- Started all beams from <s> rather than from <pad> (line 40)
"""

import torch
import data

class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), data.START_ID, dtype=torch.long, device=device)]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob, ngram_block_size=None):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        if ngram_block_size:
            beam_lk = self.block_ngram(beam_lk, ngram_size=ngram_block_size)
        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        next_ys = best_scores_id - prev_k * num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(next_ys)

        # End condition is when top-of-beam is STOP_ID.
        if self.next_ys[-1][0].item() == data.STOP_ID:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[data.START_ID] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k, without_stop_id=False):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            word_idx = self.next_ys[j+1][k]
            hyp.append(word_idx)
            if without_stop_id and word_idx == data.STOP_ID:
                hyp.clear()
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

    def block_ngram(self, word_scores, ngram_size):
        """ Mask words which make up a repeated n-gram """

        for k in range(self.size):
            hyp = self.get_hypothesis(k)
            old_gram = hyp[len(hyp)-ngram_size+1:]

            for i in range(len(hyp)-ngram_size+1):
                if hyp[i:i+ngram_size-1] == old_gram:
                    next = hyp[i+ngram_size-1]
                    word_scores[k][next] = float("-inf")

        return word_scores
