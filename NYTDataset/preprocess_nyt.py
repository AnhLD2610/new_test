import os
import time
import tarfile
import re
import sys
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from multiprocessing import Pool
from config import config
from utils import format_example, get_formatted_tokens

root_dir = os.path.expanduser(config.nyt_dataset_path + "/data/")
n_batches = 100
percent_spit = (0.9, 0.05, 0.05) # train, validation, test
np.random.seed(0)
short = False # load full summary/article

VOCAB_SIZE = 200000


def clean_meta(txt):
    possible_meta_tags = ['photo', 'chart', 'map', 'graph', 'drawing', 'diagram', 'list']
    labels = "(?:\s*(?:" + '|'.join(["(?:" + x + "s?)" for x in possible_meta_tags]) + ")\s*;?)*"
    pattern = ';?' + labels + '\s*(?:\([^)]*\).?)*\s*$'
    return re.subn(pattern, '.', txt, count=1)


def clean_batch(batch, article=False):
    cleaned_batch = []
    vocab_counter = Counter()
    for elem in batch:
        txt, n_subs = clean_meta(elem) # remove ending meta data
        joined_ex, tokens = get_formatted_tokens(txt)
        vocab_counter.update(tokens)
        cleaned_batch.append(joined_ex)
    return cleaned_batch, vocab_counter


def strip_html(node):
    return ' '.join(node.findAll(text=True)).replace("\n", " ")


def parse_body(parsed_html):
    body = parsed_html.findAll('body.content')[0]
    body_class = "lead_paragraph" if short else "full_text"
    return strip_html(body.findAll("block", {"class": body_class})[0])


def parse_abstracts(parsed_html):
    abstracts = parsed_html.findAll("abstract")
    if short:
        headlines = parsed_html.findAll('hedline')
        abstracts = [] if len(headlines) == 0 else headlines[0].findAll("hl1")
    return abstracts


def parse_html(html):
    parsed_html = BeautifulSoup(html, features="html.parser")
    abstracts = parse_abstracts(parsed_html)
    if len(abstracts) > 0:
        try:
            abstract = strip_html(abstracts[0])
            body_text = parse_body(parsed_html)
            return (abstract, body_text)
        except:
            return None
    return None

def read_tar(path):
    articles = []
    summaries = []
    tar = tarfile.open(path, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            content = f.read()
            parsed = parse_html(content)
            if parsed is not None:
                summary, article = parsed
                articles.append(article)
                summaries.append(summary)
    return summaries, articles


def write_chunk(index, summaries, articles):
    vocab_counter = Counter()
    if index % 20 == 0:
        directory = "test"
    elif index % 20 == 1:
        directory = "validation"
    else:
        directory = "train"

    filenames = [root_dir + "{}/{}-{}.txt".format(directory, x, index) for x in ["summary", "article"]]

    summaries, sum_counts = clean_batch(summaries)
    articles, art_counts = clean_batch(articles)

    for filename,batch in zip(filenames, [summaries, articles]):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for entry in batch:
                f.write("%s\n" % entry)

    if directory == "train":
        vocab_counter = sum_counts + art_counts

    return vocab_counter


def process_file(chunk_index, path):
    print("Processing " + path)
    summaries, articles = read_tar(path)
    vocab_counter = write_chunk(chunk_index,summaries,articles)
    return vocab_counter, len(summaries)


def get_filepaths():
    final = []
    for root, directories, filenames in os.walk(root_dir, topdown=True):
        directories[:] = sorted([d for d in directories if d >= "1996"])
        final += [os.path.join(root,fl) for fl in sorted(filenames) if ".tgz" in fl]
    return final


if __name__ == '__main__':
    start = time.time()

    if len(sys.argv) > 1:
        short = sys.argv[1] == "-s" # load headline and first paragraph

    ## Process Files
    filepaths = get_filepaths()
    out_dir = "preprocessed-small/" if short else "preprocessed/"
    root_dir += out_dir

    pool=Pool()
    results = pool.starmap(process_file, enumerate(filepaths))
    pool.close()
    pool.join()

    vocab_counters, n_items = zip(*results)
    final_vocab_counter = sum(vocab_counters, Counter())
    final_item_count = sum(n_items)

    # Adapted from See et. al
    with open(root_dir + "vocab.txt", 'w') as f:
        for word, count in final_vocab_counter.most_common(VOCAB_SIZE):
            f.write(word + ' ' + str(count) + '\n')

    end = time.time()

    print("Saved " + str(final_item_count) + " summary and article pairs.")
    print("Took " + str(end - start) + "s")


