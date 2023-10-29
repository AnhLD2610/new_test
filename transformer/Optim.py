#!/usr/bin/env python
"""
File: transformer
Date: 2/15/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Modified from:
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py

Our modifications:
- Removed function zero_grad
- The original ScheduledOptim class is now ScheduledOptim_V0
- Modified learning rate scale in new ScheduledOptim class (lines 33-37)
"""

import numpy as np

class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, d_model, n_warmup_steps, base_lr=None):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.base_lr = base_lr or np.power(d_model, -0.5)
        self.lr = 0.0

    def _get_lr_scale(self):
 
        if self.n_current_steps < self.n_warmup_steps:
            return np.power(self.n_warmup_steps, -0.5)

        x = self.n_current_steps - self.n_warmup_steps
        return np.power(self.n_current_steps, -0.5)

    def update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_current_steps += 1

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    @property
    def learning_rate(self):
        return self.base_lr * self._get_lr_scale()


class ScheduledOptim_V0:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, d_model, n_warmup_steps, base_lr=None):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.base_lr = base_lr or np.power(d_model, -0.5)
        self.lr = 0.0

    def _get_lr_scale(self):      
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_current_steps += 1

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    @property
    def learning_rate(self):
        return self.base_lr * self._get_lr_scale()


