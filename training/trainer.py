#!/usr/bin/env python
"""
File: train
Date: 2/9/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Based on:
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/train.py
"""

import os, sys
import time
from tqdm import tqdm
import logging
logger = logging.getLogger()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils
from utils.tensorboard import Tensorboard
from models import HyperParameters
from torch.utils.data import Dataset, DataLoader
from evaluation.evaluator import Evaluator
from transformer.Optim import ScheduledOptim
from data.batch import Batch

class Trainer(object):

    def __init__(self, vocab, hyperams: HyperParameters, output_dir=None, device=torch.device('cpu')):

        self.vocab = vocab
        self.hyperams = hyperams
        self.output_dir = output_dir
        self.device = device

        # these are initialized in setup_training
        self.train_loader = None
        self.val_loader = None

        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.optimizer = None
        self.optimizer_state = None
        self.validator = None

        self.train_dir = None
        self.model_dir = None
        self.tensorboard = None

    def restore(self, model_state):
        self.optimizer_state = model_state['optimizer']

    def save_model(self, loss, epoch):
        state = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'current_loss': loss,
            **self.model.get_state()
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (epoch, int(time.time())))
        torch.save(state, model_save_path)
        utils.prune_model_files(self.model_dir)

    def setup_training(self, model, train_dataset, val_dataset, name=None):
        self.model = model
        self.train_dataset = train_dataset
        self.validate = val_dataset is not None
        self.val_dataset = val_dataset
        self.name = name

        # Adam Optimizer initialization
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params,
                                    self.hyperams.learning_rate,
                                    betas=self.hyperams.adam_betas, eps=self.hyperams.adam_eps)

        if self.optimizer_state is not None:
            self.restore_optimizer(self.optimizer_state)

        if self.hyperams.schedule_lr:
            self.lr_scheduler = ScheduledOptim(self.optimizer, self.hyperams.d_model, self.hyperams.warmup_steps,
                                               base_lr=self.hyperams.base_learning_rate)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.hyperams.batch_size, shuffle=True)
        if self.validate:
            self.validator = Evaluator(self.model, val_dataset, self.vocab, self.hyperams,
                                       model_dir=self.output_dir, device=self.device)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.hyperams.batch_size, shuffle=True)

        self.step = 1

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

            # the directory to put *all* the training files for this one model
            training_dir = name or 'model_%d' % (int(time.time()))
            self.train_dir = os.path.join(self.output_dir, training_dir)

            os.makedirs(self.train_dir, exist_ok=True)

            hp_file = os.path.join(self.train_dir, "hyperparameters.json")
            logger.debug("Saving hyper-parameters to: %s" % hp_file)
            self.hyperams.save(hp_file)

            # the directory to put just the saved/trained hyperparameters
            self.model_dir = os.path.join(self.train_dir, 'model')
            os.makedirs(self.model_dir, exist_ok=True)

            self.save_command_line_args()
            self.tensorboard = Tensorboard(self.train_dir)

    def restore_optimizer(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        from copy import deepcopy
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        # Update the state
        from itertools import chain
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        import collections
        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, collections.abc.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        from collections import defaultdict
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.optimizer.__setstate__({'state': state, 'param_groups': param_groups})

    def train_one_batch(self, preprocessed_batch):
        src, tgt = preprocessed_batch

        self.optimizer.zero_grad()

        preds, log_dict = self.model(src, tgt, log_dict=True)
        gold = self.model.gold(src, tgt)
        loss, num_correct, ratio_correct = self.cal_performance(preds, gold, smoothing=False)

        if self.hyperams.is_coverage:
            # loss = (1 - self.hyperams.lr_coverage) * loss + self.hyperams.lr_coverage * self.model.coverage_loss
            loss += self.hyperams.lr_coverage * self.model.coverage_loss

        loss.backward()

        # clip gradient
        if self.hyperams.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperams.grad_clip_norm)

        # udpate the parameters and learning rate
        self.optimizer.step()
        if self.hyperams.schedule_lr:
            self.lr_scheduler.update_learning_rate()

        return loss, num_correct, ratio_correct, log_dict

    def preprocess(self, batch):
        source_batch, target_batch = batch
        src_batch, tgt_batch = zip(*sorted(zip(source_batch, target_batch),
                                           key=lambda e: len(e[0]),
                                           reverse=True))
        src = Batch(src_batch, max_seq_len=self.hyperams.max_seq_len_src, vocab=self.vocab, device=self.device)
        tgt = Batch(tgt_batch, max_seq_len=self.hyperams.max_seq_len_tgt, vocab=self.vocab, device=self.device)

        return src, tgt

    def train(self, model, train_dataset: Dataset, val_dataset=None, epochs=1, name=None, val_every=1):
        self.setup_training(model, train_dataset, val_dataset, name=name)

        for epoch in range(epochs):
            logger.info("Epoch %d" % epoch)
            self.model.train() # set the model to "training" mode

            iterator = tqdm(self.train_loader, unit='Batch')
            for i_batch, batch in enumerate(iterator):

                preprocessed_batch = self.preprocess(batch)
                loss, num_correct, ratio_correct, log_dict = self.train_one_batch(preprocessed_batch)

                self.tensorboard.log_scalar("train/loss", float(loss), self.step)
                self.tensorboard.log_scalar("train/NumCorrect", float(num_correct), self.step)
                self.tensorboard.log_scalar("train/RatioCorrect", float(ratio_correct), self.step)
                self.tensorboard.log_scalar("train/Learning_Rate", self.current_learning_rate(), self.step)
                for k,v in log_dict.items():
                    self.tensorboard.log_scalar(f"model/{k}", float(v), self.step)
                self.step += 1

                iterator.set_description("loss: %.3f, %.1f%% correct" % (loss, ratio_correct*100))

            self.tensorboard.flush()
            self.save_model(loss, epoch)

            if self.validate and (epoch % val_every == 0):
                logger.info("Validating...")
                result , _ = self.validator.evaluate(num_batches=4)
                logger.info("Validation result")
                logger.info(utils.format_result(result))
                self.log_validation(result, epoch=epoch)

    def log_validation(self, result, epoch=None):
        """ logs the validation results to TensorBoard """
        for rouge_type, score in result.items():
            self.tensorboard.log_scalar(rouge_type + "/precision", score.mid.precision, epoch)
            self.tensorboard.log_scalar(rouge_type + "/recall", score.mid.recall, epoch)
            self.tensorboard.log_scalar(rouge_type + "/fmeasure", score.mid.fmeasure, epoch)

    def cal_performance(self, pred, gold, smoothing=False):
        """ Apply label smoothing if needed """

        loss = self.cal_loss(pred, gold, smoothing)

        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(self.vocab[self.vocab.pad_token])
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        ratio_correct = n_correct / non_pad_mask.sum().item()

        return loss, n_correct, ratio_correct

    def cal_loss(self, pred, gold, smoothing):
        """ Calculate cross entropy loss, apply label smoothing if needed """
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.vocab[self.vocab.pad_token])
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.nll_loss(pred, gold, ignore_index=self.vocab[self.vocab.pad_token], reduction='sum')

        return loss

    def current_learning_rate(self):
        """ returns the current learning rate """
        if self.hyperams.schedule_lr:
            return self.lr_scheduler.learning_rate
        return self.hyperams.learning_rate

    def save_command_line_args(self):
        args_file = os.path.join(self.train_dir, "args.txt")
        with open(args_file, 'w') as f:
            f.write(" ".join(sys.argv))
