#!/usr/bin/env python
"""
File: hyperparameters
Date: 2/9/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import json


class HyperParameters(object):
    """ Simple class for storing model hyper-parameters """

    def __init__(self, use_wmt=False):
        self.seed = 42
        self.epsilon = 1e-300
        self.model_type = "pointer"  # toggle pointer generator

        self.batch_size = 4
        self.num_epochs = 64
        self.grad_clip_norm = 1.0

        self.vocab_size = 30000  # zero means use entire provided vocab
        self.num_training_examples = 40000

        self.beam_size = 4
        self.n_best_hypotheses = 1
        self.ngram_block_size = None # set to None to turn off ngram blocking

        # Sequence length limit parameters
        self.max_seq_len = None
        self.max_seq_len_src = 400
        self.max_seq_len_tgt = 128

        # Word Embedding parameters
        self.emb_dim = 128
        self.use_pretrained_embeddings = True
        self.pretrained_emb_dim = 300
        self.freeze_embeddings = True

        # Transformer parameters
        self.transformer_layers = 1
        self.d_model = 512
        self.d_inner = 2048
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
        self.dropout = 0.1
        self.tgt_emb_prj_weight_sharing = True
        self.emb_src_tgt_weight_sharing = True

        # Adam Optimization parameters
        self.learning_rate = 0.01

        self.schedule_lr = True
        self.base_learning_rate = 0.03
        self.adam_betas = (0.9, 0.98)
        self.adam_eps = 1e-9
        self.warmup_steps = 900

        # ===================== not really using the ones below here

        # from See. et al
        self.hidden_dim = 256
        self.lstm_layers = 1
        self.bidir = True

        # initialization parameters (from atulkum/pointer_summarizer)
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4

        self.max_grad_norm = 2.0
        self.adagrad_init_acc = 0.1

        self.max_enc_steps = 400
        self.max_dec_steps = 100

        # Coverage related parameters
        self.is_coverage = True
        self.cov_loss_wt = 1.0
        self.eps = 1e-12
        self.lr_coverage = 1.0

        if use_wmt:
            self.set_to_wmt_params()

    def override(self, params):
        """
        Overrides attributes of this object with those of "params".
        All attributes of "params" which are also attributes of this object will be set
        to the values found in "params". This is particularly useful for over-riding
        hyperparamers using those set from the command-line arguments
        :param settings: Object with attributes to override in this object
        :return: None
        """
        for attr in vars(params):
            if hasattr(self, attr) and getattr(params, attr) is not None:
                value = getattr(params, attr)
                setattr(self, attr, value)

    def save(self, file):
        """ save the hyper-parameters to file in JSON format """
        with open(file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def restore(file):
        """ restore hyper-parameters from JSON file """
        with open(file, 'r') as f:
            data = json.load(f)
        hp = HyperParameters()
        hp.__dict__.update(data)
        return hp


    def set_to_wmt_params(self):
        self.transformer_layers = 1 # they used 6
        self.emb_dim = 512
        self.n_head = 8
        self.dropout = 0.1
        self.d_inner = 2048

        self.schedule_lr = True
        self.base_learning_rate = 2 # ??
        self.adam_betas = (0.9, 0.998)
        self.warmup_steps = 8000

        self.num_epochs = 64 # they put 200,000...
        # also use accum_count 2, seemed maybe useful but i couldn't figure out what it was
