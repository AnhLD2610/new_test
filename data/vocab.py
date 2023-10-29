#!/usr/bin/env python
"""
File: vocab
Translated from : https://github.com/abisee/pointer-generator/blob/master/data.py
"""

import data
import csv
from collections import Counter
from itertools import chain
from typing import List
import torch
from utils import cast_to_string

class Vocab:
    def __init__(self, vocab_file, max_size=0):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in (self.unk_token, self.pad_token, self.start_decoding, self.stop_decoding):
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'rb') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print("Warning: incorrectly formatted line in vocabulary file: %s\n" % line)
                    continue
                w = pieces[0].decode("utf-8")
                if w in [data.SENTENCE_START, data.SENTENCE_END, data.UNKNOWN_TOKEN,
                         data.PAD_TOKEN, data.START_DECODING, data.STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    @property
    def pad_token(self):
        return data.PAD_TOKEN

    @property
    def unk_token(self):
        return data.UNKNOWN_TOKEN

    @property
    def start_decoding(self):
        return data.START_DECODING

    @property
    def stop_decoding(self):
        return data.STOP_DECODING

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        word = cast_to_string(word)
        if word not in self._word_to_id:
            return self._word_to_id[data.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id, oovs=[]):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            oov_idx = word_id - len(self._id_to_word)
            if oov_idx in range(len(oovs)):
                return cast_to_string(oovs[oov_idx])
            else:
                raise ValueError('Id not found in vocab: %d' % word_id)
        return cast_to_string(self._id_to_word[word_id])

    @property
    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return self.size

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id(word)

    def __setitem__(self, key, value):
        # Raise error, if one tries to edit the VocabEntry.
        raise ValueError('vocabulary is readonly')

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self._word_to_id

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self._word_to_id[word] = len(self)
            self._id_to_word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        @param sents (list[str] or list[list[str]]): sentence(s) in words
        @return word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if type(sents[0]) == list:
            return [self.words2indices(s) for s in sents]
        else:
            return [self.word2id(w) for w in sents]

    def indices2words(self, word_ids, oovs=[]):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word(w_id, oovs=oovs) for w_id in word_ids]

    def to_input_tensor(self, sents: List[str], device: torch.device,
                        seq_len=None, with_position=True):
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        tokenized = [sent.split() for sent in sents]

        if with_position:
            padded_sents, pos = pad_sents(tokenized, data.PAD_TOKEN, length=seq_len, with_position=with_position)
            pos_tensor = torch.tensor(pos, dtype=torch.long, device=device)

        else:
            padded_sents = pad_sents(tokenized, data.PAD_TOKEN, length=seq_len, with_position=with_position)

        word_ids = self.words2indices(padded_sents)
        sents_var = torch.tensor(word_ids, dtype=torch.long, device=device)

        if with_position:
            return sents_var, pos_tensor
        else:
            return sents_var

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz

        Args:
          fpath: place to write the metadata file
        """
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(len(self)):
                writer.writerow({"word": self._id_to_word[i]})

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
      article_words: list of words (strings)
      vocab: Vocabulary object

    Returns:
      ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number.
        If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers
         will be 50000, 50001, 50002.
      oovs:
        A list of the OOV words in the article (strings), in the order corresponding to their
         temporary article OOV numbers."""
    ids = list()
    oovs = list()
    unk_id = vocab.word2id(data.UNKNOWN_TOKEN)
    for w in article_words:
        word_id = vocab.word2id(w)
        if word_id == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(len(vocab) + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(word_id)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
    Args:
      abstract_words: list of words (strings)
      vocab: Vocabulary object
      article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers
    Returns:
      ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(data.UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = len(vocab) + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def pad_sents(sents, pad_token, length=None, with_position=False):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """

    sents_padded = list()
    max_len = length or max(map(len, sents))

    if with_position:
        pos = list()

    for sent in sents:
        sent = [data.START_DECODING] + sent + [data.STOP_DECODING]
        original_length = len(sent)
        final_length = min(original_length, max_len)

        if original_length > max_len: # too long?
            sent = sent[:max_len]  # clip
            num_pads = 0

        else: # too short?
            num_pads = max_len - original_length
            padding = [pad_token] * num_pads
            sent = sent + padding

        assert(len(sent) == max_len)

        if with_position:
            sent_pos = list(range(1, 1 + final_length))
            sent_pos = sent_pos + [0] * num_pads
            pos.append(sent_pos)

            assert(len(sent_pos) == len(sent))

        sents_padded.append(sent)

    if with_position:
        return sents_padded, pos
    else:
        return sents_padded
