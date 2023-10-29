#!/usr/bin/env python
"""
File: run
Date: 2/27/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os, sys
import argparse, logging
import torch

from config import config
from models import HyperParameters
from models.model_types import get_model_class

from data.vocab import Vocab
from data.glove import Glove
from data.dataset_types import get_dataset_class
from data.dataset_groups import DatasetGroup
from evaluation.evaluator import Evaluator
from utils import decide_cuda, set_seed, load_model_state, create_dir_at


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        logger.error("Could not find model at: %s" % args.model)
        return

    hp_file = os.path.join(args.model, "hyperparameters.json")
    if not os.path.exists(hp_file):
        logger.error("Could not find hyperparameters at: %s" % hp_file)
        return
    logger.debug("Restoring hyperparameters from %s" % hp_file)
    hyperams = HyperParameters.restore(hp_file)
    hyperams.override(args)

    use_cuda = decide_cuda(logger, args, config)
    device = torch.device('cuda' if use_cuda else 'cpu')
    set_seed(hyperams.seed, use_cuda=use_cuda)

    ############################
    # Load Dataset/Embeddings
    ############################
    dataset_paths, Dataset = get_dataset_class(args.dataset)
    if dataset_paths.path is None or os.path.exists(dataset_paths.path):
        logger.info("Data set location: %s" % dataset_paths.path)
    else:
        logger.error("Data set not found at %s. Did you fix config/config.ini?" % dataset_paths.path)
        return

    test_dataset = Dataset(dataset_paths, group=DatasetGroup.test, num_chunks=args.chunks)

    logger.debug("Loading vocab...")
    vocab = Vocab(dataset_paths.vocab_path, max_size=hyperams.vocab_size)

    embeddings = None
    if hyperams.use_pretrained_embeddings:
        logger.debug("Loading emeddings...")
        emb_file = config.glove_emb_path
        glove = Glove(emb_file, vocab, emb_dim=hyperams.pretrained_emb_dim)
        hyperams.d_model = glove.emb_dim
        embeddings = glove.get_embedding_table()

    ############################
    # Load Model
    ############################

    logger.info("Instantiating model of type %s" % hyperams.model_type)
    Model = get_model_class(hyperams.model_type)
    model = Model(hyperams, vocab, word_embeddings=embeddings, device=device, compute_extra=args.compute_extra)
    if use_cuda:
        model.cuda()

    model_path, model_name, model_state = load_model_state(args.model, model.device, model_num=args.model_num)
    logger.info("Loading model from file %s" % model_path)
    model.load_state_dict(model_state)
    logger.info("Model instantiated.")

    ############################
    # Evaluate
    ############################
    logger.debug("Instantiating evaluator...")
    evaluator = Evaluator(model, test_dataset, vocab, hyperams, model_dir=args.model, device=device)
    logger.debug("Evaluator instantiated.")

    output_dir = create_dir_at(args.model, model_name, remove=False)

    if args.ngram_blocking_experiment_num:
        evaluator.eval_n_gram_blocking(num_batches=args.num_batches, max_block=args.ngram_blocking_experiment_num, output_dir=output_dir)
    else:
        logger.info("Starting evaluation...")
        results, result_path = evaluator.evaluate(num_batches=args.num_batches, write_to_file=True, output_dir=output_dir)
        logger.debug("Evaluation complete. Saved results to %s" % result_path)

    logger.debug("Exiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Test summarization model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    dataset_options = parser.add_argument_group("Dataset")
    dataset_options.add_argument("-dataset", "--dataset", choices=["cnn", "nyt", "tiny", "giga"], default="cnn",
                                 help="The data set to use for training")
    dataset_options.add_argument("--batches", dest="num_batches", type=int, help="Number of batches to test on")
    dataset_options.add_argument('--path', help="Data set input file")
    dataset_options.add_argument('--chunks', type=int, help="Number of chunks to use")

    model_options = parser.add_argument_group("Model")
    model_options.add_argument("--model-type", dest="model_type",
                               choices=["baseline", "pointer"], default=None,
                               help="The type of model to train")
    model_options.add_argument("--model", required=True, help="Model path to restore model from and save results")
    model_options.add_argument("--model-num", required=False, dest="model_num", type=int, help="Model epoch num to restore")

    model_options.add_argument("--pointer", dest="pointing", action="store_true", help="Use pointer generator mechanism")
    model_options.add_argument("--no-pointer", dest="pointing", action="store_false", help="Disable pointer generator mechanism")
    parser.set_defaults(pointing=None)

    hyperams_options = parser.add_argument_group("HyperParameters")
    hyperams_options.add_argument("-beam", "--beam-size", type=int, dest='beam_size', help="Beam size of decoding")
    hyperams_options.add_argument('--blocking', dest="ngram_block_size", type=int,
                                  help="Number of ngrams to block during decoding")
    hyperams_options.add_argument('--blocking-experiment', dest="ngram_blocking_experiment_num", type=int, default=None,
                                  help="Run an experiment on n gram blocking up where blocking = 1 up to the specifid value")
    hyperams_options.add_argument('--compute-extra-info', dest="compute_extra", action="store_true",
                                  help="Compute extra info for output during decoding (attention info for pointer models)")

    gpu_options = parser.add_argument_group("GPU")
    gpu_options.add_argument("-gpu", "--gpu", action='store_true', help="Enable GPU")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="INFO", help="Logging level")
    args = parser.parse_args()

    # Setup the logger
    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


if __name__ == "__main__":
    main()

