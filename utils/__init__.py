"""
File: __init__.py
Date: 2/19/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""
import torch
import os
import re
import string
import random
import shutil
from itertools import chain

def decide_cuda(logger, args, config):
    """ decides whether to use CUDA based on the command line arguments and configuration """
    use_cuda = False
    if args.gpu or config.use_cuda:
        if torch.cuda.is_available():
            use_cuda = True
            logger.debug("Using GPU, %i device(s)" % torch.cuda.device_count())
        else:
            logger.warning("GPU unavailable")
    return use_cuda

def set_seed(seed, use_cuda=False):
    """ Sets random seeds for reproduciblity """
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def create_dir_at(base_path, dir_name, remove=False):
    dir_path = os.path.join(base_path, dir_name)
    if remove and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def format_result(result):
    return "\n\t"+"\n\t".join("{}\t{}".format(k, v.mid) for k, v in result.items())

def get_model_epoch(filename):
    return int(re.match(r'model_(\d+)_\d+', filename).group(1))

def sort_model_files(model_files):
    indexed_files = map(lambda f: (get_model_epoch(f), f), model_files)
    sorted_indexed_files = sorted(indexed_files, key=lambda x: x[0])
    return list(map(lambda f: f[1], sorted_indexed_files))

def load_model_state(model_path, device, model_num=None):
    model_dir = os.path.join(model_path, "model")
    model_files = os.listdir(model_dir)
    if model_num is not None:
        selected_file = [f for f in model_files if get_model_epoch(f) == model_num][0]
    else:
        selected_file = sort_model_files(model_files)[-1]
    model_file = os.path.join(model_dir, selected_file)
    return model_file, selected_file, torch.load(model_file, map_location=device)


def prune_model_files(model_dir, n_to_save=5):
    saved_models = sort_model_files(os.listdir(model_dir))
    n_to_remove = len(saved_models) - n_to_save
    if n_to_remove > 0:
        to_remove = saved_models[:n_to_remove]
        for r in to_remove:
            os.remove(os.path.join(model_dir,r))
    return max(n_to_remove,0)


def cast_to_string(word):
    if type(word) == str:
        return word
    else:
        decoded = word.decode("utf-8")
        return decoded if type(decoded) == str else ""


def get_formatted_tokens(text):
    def token_split_punc(token):
        escaped_punc = '([' + re.escape(string.punctuation) + ']*)'
        pattern = escaped_punc + '(.*?)' + escaped_punc
        matches = re.fullmatch(pattern, token)
        return list(filter(None, list(matches.groups())))

    tokens = text.split(" ")
    tokens = list(filter(None, [t.lower().strip() for t in tokens]))  # strip, remove empty
    tokens = list(chain(*list(map(token_split_punc, tokens))))

    return ' '.join(tokens), tokens


def format_example(text):
    """
    :param text: example text
    :return: formatted example text, list of tokens
    """
    return get_formatted_tokens(text)[0]


