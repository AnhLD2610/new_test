#!/usr/bin/env python
"""
File: evaluate
Date: 2/27/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Based on:
https://github.com/atulkum/pointer_summarizer/blob/master/training_ptr_gen/train.py
"""

import os
import json
from tqdm import tqdm
import torch
import logging
logger = logging.getLogger()

import utils
from utils.tensorboard import Tensorboard
from models import HyperParameters
from torch.utils.data import Dataset, DataLoader
from models.summerizer import TransformerSummarizer
from data.dataset_types import DatasetType, get_dataset_class
from data.dataset_groups import DatasetGroup
from data.batch import Batch

from evaluation.rouge.rouge_scorer import RougeScorer
from evaluation.rouge.rouge_scoring import BootstrapAggregator


class Evaluator(object):
    def __init__(self, model,
                 dataset: Dataset,
                 vocab,
                 hyperams: HyperParameters,
                 model_dir=None,
                 device=torch.device('cpu'),
                 use_stemmer=True):

        self.model = model
        self.vocab = vocab

        self.hyperams = hyperams
        self.model_dir = model_dir
        self.device = device
        self.use_stemmer = use_stemmer

        self.dataset = dataset
        self.loader = None

        self.summarizer = TransformerSummarizer(
            self.model,
            self.vocab,
            hyperams.max_seq_len_tgt,
            beam_size=hyperams.beam_size,
            num_best=hyperams.n_best_hypotheses,
            n_gram_block_size=hyperams.ngram_block_size,
            device=device)

        self.rouge_types = ["rouge1", "rouge2", "rougeL"]
        self.scorer = RougeScorer(self.rouge_types, self.use_stemmer)
        self.aggregator = BootstrapAggregator()

    def setup_eval(self):
        self.loader = DataLoader(self.dataset, batch_size=self.hyperams.batch_size, shuffle=True)

    def eval_one_batch(self, preprocessed_batch, aggregate=True):
        src, target_batch = preprocessed_batch

        batch_hyp, batch_scores, extra_info = self.summarizer.summarize_batch(src)

        decoded_results = []
        for i in range(len(batch_hyp)):
            target = target_batch[i]
            if type(target) == bytes:
                target = target.decode("utf-8")
            score = self.scorer.score(target, batch_hyp[i])
            if aggregate:
                self.aggregator.add_scores(score)
            output = (target, batch_hyp[i]) if extra_info is None else (target, batch_hyp[i], extra_info[i])
            decoded_results += [output]

        return decoded_results


    def preprocess(self, batch):
        source_batch, target_batch = batch
        src_batch, tgt_batch = zip(*sorted(zip(source_batch, target_batch),
                                           key=lambda e: len(e[0]),
                                           reverse=True))
        src = Batch(src_batch, max_seq_len=self.hyperams.max_seq_len_src, vocab=self.vocab, device=self.device)

        return src, tgt_batch

    def evaluate(self, num_batches=None, write_to_file=False, output_dir=None):
        """ runs the evaluation on the validation set for a single epoch

        :param num_batches: The number of batches to run through evaluation (defaults to None
        which means use all the batches in the data set.
        :param write_to_file: Whether to write the results to file or just print them
        :return: The results of the evaluation (ROUGE scores)
        """
        output_dir = output_dir or self.model_dir
        output_path = os.path.join(output_dir, "test_output.txt")
        metric_path = os.path.join(output_dir, "test_results.json")
        result = None

        self.setup_eval()
        self.model.eval()
        self.reset()
        if write_to_file:
            utils.remove_file_if_exists(output_path)

        num_batches = num_batches or len(self.loader)
        for batch_i in tqdm(range(num_batches), unit="Batch"):
            batch = next(iter(self.loader))
            preprocessed_batch = self.preprocess(batch)

            decoded_output = self.eval_one_batch(preprocessed_batch)

            if write_to_file:
                self.write_decoded_output(decoded_output, output_path)
                result = self.aggregator.aggregate()
                self.write_metrics(result, metric_path)

        if not result:
            result = self.aggregator.aggregate()

        return result, metric_path

    def write_decoded_output(self, decoded_output, output_path):
        with open(output_path, "a") as f:
            for output in decoded_output:
                f.write(str(output) + "\n")

    def write_metrics(self, result, metric_path):
        with open(metric_path, 'w') as f:
            json.dump(result, f, indent=4)

    def reset(self):
        self.aggregator = BootstrapAggregator()

    def eval_n_gram_blocking(self, num_batches=None, max_block=5, output_dir=None):

        output_dir = output_dir or self.model_dir
        output_dir_name = "n-gram-blocking"
        ngram_path = utils.create_dir_at(output_dir, output_dir_name, remove=True)

        tensorboard = Tensorboard(output_dir)

        for i in range(1, max_block+1):
            logger.info("Evaluating with n-gram blocking of %d" % i)

            block_dir_path = utils.create_dir_at(ngram_path, f"block_{i}")

            self.summarizer.set_ngram_block_size(i)
            result, _ = self.evaluate(num_batches=num_batches, write_to_file=True, output_dir=block_dir_path)

            for rouge_type, score in result.items():
                base = f"{output_dir_name}/{rouge_type}"
                tensorboard.log_scalar(base + "/precision", score.mid.precision, i)
                tensorboard.log_scalar(base + "/recall", score.mid.recall, i)
                tensorboard.log_scalar(base + "/fmeasure", score.mid.fmeasure, i)

        tensorboard.flush()

    def test(dataset, num_batches=10):
        rouge_types = ["rouge1", "rouge2", "rougeL"]
        scorer = RougeScorer(rouge_types, True)
        aggregator = BootstrapAggregator()
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        num_batches = num_batches or len(loader)
        for batch_i in tqdm(range(num_batches), unit="Batch"):
            batch = next(iter(loader))

            source_batch, target_batch = batch
            src_batch, tgt_batch = zip(*sorted(zip(source_batch, target_batch),
                                               key=lambda e: len(e[0]),
                                               reverse=True))

            for i in range(len(tgt_batch)):
                target = tgt_batch[i].decode("utf-8")
                score = scorer.score(target, target)
                aggregator.add_scores(score)
        result = aggregator.aggregate()
        print(result)

### TEST EVALUATOR
if __name__ == "__main__":
    dataset_type = DatasetType.cnn
    dataset_paths, Dataset = get_dataset_class(dataset_type)
    test_dataset = Dataset(dataset_paths, group=DatasetGroup.test, num_chunks=1)

    Evaluator.test(test_dataset)
