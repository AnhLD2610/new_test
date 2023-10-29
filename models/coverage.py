#!/usr/bin/env python
"""
File: train
Date: 3/9/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from data.batch import Batch
from models.beam_searcher import BeamSearcher


class Coverage(object):  # wrapper for necessary coverage vector calculations

    def __init__(self):
        self.attn_dists = []
        self.padding_mask = None

    def add_attn_dist(self, attn_dists):
        batch_size, _ = attn_dists.shape
        self.attn_dists.append(attn_dists)
        self.padding_mask = torch.ones(batch_size, len(self.attn_dists))

    def _mask_and_avg(self, values, padding_mask):
      """
      Applies mask to values then returns overall average (a scalar)
      Args:
        values: a list length max_dec_steps containing arrays shape (batch_size).
        padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
      Returns:
        a scalar
      """

      dec_lens = torch.sum(padding_mask, 1) # shape batch_size. float32
      values_per_ex = sum(values)/dec_lens # shape (batch_size); normalized value for each batch member
      return torch.sum(values_per_ex) # overall average


    def _coverage_loss(self):
      """
      Calculates the coverage loss from the attention distributions.
      Args:
        attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
        padding_mask: shape (batch_size, max_dec_steps).
      Returns:
        coverage_loss: scalar
      """
      if len(self.attn_dists) == 0:
          return float("inf")
      coverage = torch.zeros_like(self.attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
      covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
      for a in self.attn_dists:
        covloss = torch.sum(torch.min(a, coverage), 1) # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a # update the coverage vector
      coverage_loss = self._mask_and_avg(covlosses, self.padding_mask)
      print("Coverage Loss: ", coverage_loss.item())
      return coverage_loss
