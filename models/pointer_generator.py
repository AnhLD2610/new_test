#!/usr/bin/env python
"""
File: transformer
Date: 3/3/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Based on:
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/model.py
https://github.com/abisee/pointer-generator/blob/master/attention_decoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data

class PointerGeneratorNetwork(nn.Module):

    def __init__(self, hidden_size, context_size, device=torch.device('cpu'), epsilon=1e-30):
        super(PointerGeneratorNetwork, self).__init__()
        self.hidden_size = hidden_size # emb_dim
        self.device = device
        self.epsilon = epsilon

        self.hidden_proj = nn.Linear(hidden_size, 1, bias=False)
        self.context_proj = nn.Linear(context_size, 1, bias=False)
        self.input_proj = nn.Linear(1, 1, bias=True)
        self.cpy_switch = Variable(torch.tensor(0.01), requires_grad=True).to(device)

        nn.init.xavier_normal_(self.hidden_proj.weight, gain=2)
        nn.init.xavier_normal_(self.context_proj.weight, gain=2)
        nn.init.xavier_normal_(self.input_proj.weight, gain=2)

    def calc_pgen(self, input, context, hidden, decode=False):
        hidden_reshaped = hidden.view(-1, self.hidden_size)  # batch*(tgt_len\beam_size) x emb_dim
        h_proj = self.hidden_proj(hidden_reshaped)  # batch*(tgt_len|beam_size) x 1
        c_proj = self.context_proj(context)

        # get the input in to the right shape for x_proj
        input_reshaped = torch.unsqueeze(input[:, -1], dim=1) if decode else input.contiguous().view(-1, 1)

        x_proj = self.input_proj(input_reshaped.float())
        pgen = torch.sigmoid(h_proj + c_proj + x_proj)
        return pgen

    def forward(self, input, context, hidden, vocab_logits, attn_dist, oov_data, decode=False):
        _, tgt_len = input.shape
        pgen = self.calc_pgen(input, context, hidden, decode=decode)

        batch_with_oov_ids, _, extra_zeros = oov_data

        vocab_dist_ = pgen * F.softmax(vocab_logits, dim=1)
        attn_dist_ = (1 - pgen) * attn_dist

        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

        final_dist = vocab_dist_.scatter_add(1, batch_with_oov_ids, attn_dist_)

        normalization_factor = 1 + final_dist.shape[1] * self.epsilon
        final_dist = (final_dist + self.epsilon) / normalization_factor
        final_dist = torch.log(final_dist)

        return final_dist, attn_dist_, pgen

class PointerAttention(nn.Module):

    def __init__(self, n_head, src_len, device=torch.device('cpu')):
        super(PointerAttention, self).__init__()
        self.n_head = n_head
        self.src_len = src_len
        self.device = device

    def get_src_padding_mask(self, src_seq, tgt_seq_len, src_seq_len, batch_size, decode=False):
        batch_pad_mask = (src_seq != data.PAD_ID).type(torch.FloatTensor).to(self.device)
        if decode:
            return batch_pad_mask
        return batch_pad_mask.repeat(1, tgt_seq_len).view(-1, src_seq_len)

    def shape_attn(self, attn, decode=False):
        """
        Shapes attn appropriately.
        :param attn: Tensor of shape batch_size*n_heads x target_seq_len x source_seq_len. If decoding, batch_size
                     is actually batch_size*beam_size.
        :param decode:
        :return: Tensor of shape batch_size*(tgt_len|beam_size) x src_len
        """
        x, tgt_seq_len, src_seq_len = attn.shape
        batch_size = int(x / self.n_head)

        # Deal with n_heads dimmension
        attn = attn.contiguous().view(-1, self.n_head, tgt_seq_len, src_seq_len)

        # sum out the multiple heads of the multi-head attention
        attn = torch.sum(attn, dim=1) # batch_size, tgt_len, src_len

        attn = attn[:, -1, :] if decode else attn.view(-1, src_seq_len)
        return attn, tgt_seq_len, src_seq_len, batch_size

    def get_attn_dist(self, attn, src_seq, decode=False):
        """

        :param attn: List of Tensors of shape batch_size*n_heads x target_seq_len x source_seq_len. If decoding, batch_size
                     is actually batch_size*beam_size.
        :param src_seq:
        :param tgt_seq:
        :return: attn_dist of shape batch_size*(tgt_len\beam_size) x src_len
        """
        attn = attn[-1] # take attention from last layer

        # batch_size*(tgt_len|beam_size) x src_len
        attn, tgt_seq_len, src_seq_len, batch_size = self.shape_attn(attn, decode=decode)

        # batch_size*(tgt_len|beam_size) x src_len
        src_padding_mask = self.get_src_padding_mask(src_seq, tgt_seq_len, src_seq_len, batch_size, decode=decode)

        attn_dist_ = F.softmax(attn, dim=1) * src_padding_mask
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        return attn_dist


    def forward(self, attn, enc_output, src_seq, decode=False):
        """
        Points to a word in the target based on attention
        :param attn: Tensor of shape batch_size*n_heads, target_seq_len, source_seq_len
        :return:
        Note: The sytax (x\y) indicates that dimension x when decode=False, and dimension y when decode=True
        """

        _, tgt_seq_len, _ = attn[-1].shape
        batch_repeats = tgt_seq_len if not decode else 1

        _, src_len, emb_dim = enc_output.shape
        enc_output_shaped = enc_output.repeat(1, batch_repeats, 1).view(-1, src_len, emb_dim) # batch_size*(tgt_len|beam_size) x src_len x emb_dim

        attn_dist = self.get_attn_dist(attn, src_seq, decode=decode) # batch_size*(tgt_len|beam_size) x src_len

        attn_dist_shaped = attn_dist.unsqueeze(dim=1) # batch_size*(tgt_len|beam_size) x 1 x src_len
        c_t = torch.squeeze(torch.bmm(attn_dist_shaped, enc_output_shaped), dim=1)  # batch_size*(tgt_len|beam_size) x emb_dim

        return attn_dist, c_t
