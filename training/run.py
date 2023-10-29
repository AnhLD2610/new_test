#!/usr/bin/env python
"""
File: run
Date: 2/10/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os, sys
import argparse, logging
import torch
logger = logging.getLogger()

from config import config
from models import HyperParameters
from models.model_types import ModelType, get_model_class

from data.vocab import Vocab
from data.glove import Glove
from data.dataset_types import DatasetType, get_dataset_class
from data.dataset_groups import DatasetGroup
from training.trainer import Trainer
from utils import decide_cuda, set_seed, load_model_state
from models.transformer_pointer import TransformerPointer



def main():
    args = parse_args()

    hyperams = HyperParameters(use_wmt=args.use_wmt)
    if args.model_file is not None:
        hp_file = os.path.join(args.model_file, "hyperparameters.json")
        logger.debug(f"Restoring hyperparameters from {hp_file}")
        hyperams = HyperParameters.restore(hp_file)
    hyperams.override(args)  # override hyper-parameters with any that were provided from command line

    use_cuda = decide_cuda(logger, args, config)
    device = torch.device('cuda' if use_cuda else 'cpu')
    set_seed(hyperams.seed, use_cuda=use_cuda)

    ############################
    # Load Datasets/Embeddings
    ############################
    dataset_paths, Dataset = get_dataset_class(args.dataset)
    if dataset_paths.path is None or os.path.exists(dataset_paths.path):
        logger.info(f"Data set location: {dataset_paths.path}")
    else:
        logger.error(f"Data set not found at {dataset_paths.path}. Did you fix config/config.ini?")
        return

    train_dataset = Dataset(dataset_paths, num_chunks=args.chunks, num_examples=hyperams.num_training_examples)
    if args.validate or args.validate_every:
        val_dataset = Dataset(dataset_paths, group=DatasetGroup.validation, num_chunks=args.chunks)
    else:
        val_dataset = None

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
    # Create Model
    ############################
    hyperams.max_seq_len_src = hyperams.max_seq_len_src or hyperams.max_seq_len or train_dataset.max_sequence_length_article
    hyperams.max_seq_len_tgt = hyperams.max_seq_len_tgt or hyperams.max_seq_len or train_dataset.max_sequence_length_summary

    logger.info(f"Instantiating model of type {hyperams.model_type}")
    Model = get_model_class(hyperams.model_type)
    model = Model(hyperams, vocab, word_embeddings=embeddings, device=device)
    if use_cuda:
        model.cuda()

    model_state = None
    if args.model_file is not None:
        logger.debug(f"Loading model from file \"{args.model_file}\"")
        model_path, model_file, model_state = load_model_state(args.model_file, model.device, model_num=args.model_num)
        trainer_state = model.load_state_dict(model_state)

    logger.debug("Model instantiated.")

    if args.freeze_transformer and hasattr(model, "freeze_transformer"):
        logger.info("Freezing transformer parameters")
        model.freeze_transformer()

    ############################
    # Train
    ############################

    logger.debug("Instantiating trainer...")
    trainer = Trainer(vocab, hyperams, output_dir=args.output, device=device)
    if model_state is not None and not args.freeze_transformer:
        trainer.restore(model_state)
    logger.debug("Trainer instantiated.")

    logger.info(f"Starting training for {hyperams.num_epochs} epochs")
    trainer.train(model, train_dataset, val_dataset=val_dataset, epochs=hyperams.num_epochs, name=args.name, val_every=args.validate_every)
    logger.debug("Training complete.")

    logger.debug("Exiting.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train summarization model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_options = parser.add_argument_group("Model")
    model_options.add_argument("-model", "--model-type", dest="model_type", choices=["baseline", "pointer"],
                               default="baseline",
                               help="The type of model to train")
    model_options.add_argument("-m", "--model-file", type=str, required=False, help="Model file to restore")
    model_options.add_argument("--model-num", required=False, dest="model_num", type=int,
                               help="Model epoch num to restore")
    model_options.add_argument("--freeze-transformer", action="store_true", help="Freeze parameters in the transformer")

    dataset_options = parser.add_argument_group("Dataset")
    dataset_options.add_argument("-dataset", "--dataset", choices=["cnn", "nyt", "tiny", "giga"], default="cnn",
                                 help="The data set to use for training")
    dataset_options.add_argument('--path', help="Data set input file")
    dataset_options.add_argument('--chunks', type=int, help="Number of chunks to use")

    output_options = parser.add_argument_group("Output")
    output_options.add_argument("--output", default="model_outputs", help="Output directory")
    output_options.add_argument("--name", help="Experiment or run name")

    hyperams_options = parser.add_argument_group("HyperParameters")
    # note: make sure that the "dest" value is exactly the same as the variable name in "Hyperparameters"
    # in order for over-riding to work correctly.
    hyperams_options.add_argument("--pointing", action="store_true", help="Use pointer-generator")
    hyperams_options.add_argument("-epochs", "--epochs", dest="num_epochs",
                                  type=int, help="Number of epochs to train")
    hyperams_options.add_argument("--batch-size", dest="batch_size", type=int, help="Training batch size")
    hyperams_options.add_argument("--max-len-src", dest="max_seq_len_src", required=False,
                                  type=int, help="Maximum sequence length for the source text.")
    hyperams_options.add_argument("--max-len-tgt", dest="max_seq_len_tgt", required=False,
                                  type=int, help="Maximum sequence length for the target text.")
    hyperams_options.add_argument("--max-len", dest="max_seq_len", required=False,
                                  type=int, help="Maximum sequence length for both the source and the target text.")
    hyperams_options.add_argument("-wmt", "--wmt", dest="use_wmt", action="store_true", help="Use WMT hyperparameters")

    training_options = parser.add_argument_group("Training")
    training_options.add_argument("--validate", action='store_true', help="Run validation after each epoch")
    training_options.add_argument("--validate-every", dest='validate_every', type=int, help="Run validation after x epochs")
    training_options.add_argument("-gpu", "--gpu", action='store_true', help="Enable GPU")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument('--log', dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default="DEBUG", help="Logging level")
    args = parser.parse_args()

    # Setup the logger

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
