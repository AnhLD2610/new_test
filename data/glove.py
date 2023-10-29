#!/usr/bin/env python
"""
File: glove.py
Date: 2/11/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Based on:
https://github.com/abisee/pointer-generator/blob/master/data.py
"""

import data
import numpy as np
import os
import torch

class Glove:
    def __init__(self, emb_dir, vocab, emb_dim=100):
        self.vocab = vocab
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim
        self.emb_dir = emb_dir
        self.encoding_table = np.zeros((len(vocab), emb_dim))

        assert(emb_dim in [50,100,200,300]) #only possible values

        self.load()

    @property
    def emb_file(self):
        file_dir = os.path.join(self.emb_dir, "GloVe", "6B")
        file = list(filter(lambda f: str(self.emb_dim) in f, os.listdir(file_dir)))[0]
        return os.path.join(file_dir, file)

    def load(self):
        untracked_ids = set(self.vocab._id_to_word.keys())
        with open(self.emb_file, 'rb') as emb_f:
            for line in emb_f:
                tokens = line.decode().split()
                word = tokens[0]
                if self.vocab.__contains__(word):
                    word_id = self.vocab.word2id(word)
                    embedding = list(map(float, tokens[1:]))
                    self.encoding_table[word_id] = embedding
                    untracked_ids.remove(word_id)
        for id in untracked_ids:
            self.add(id)
        self.encoding_table[data.PAD_ID] = 0.
        print("Finished loading GloVe embeddings. %i words not found" % len(untracked_ids))

    def id2emb(self, word_id):
        """Returns the embedding corresponding to an id (integer)."""
        if word_id not in range(len(self.encoding_table)):
            return self.encoding_table[data.UNK_ID]
            #raise ValueError('Id not found in vocab: %d' % word_id)
        return self.encoding_table[word_id]

    def get_embedding_table(self):
        return torch.FloatTensor(self.encoding_table)

    def __len__(self):
        """ Compute number of embeddings.
        @returns len (int): number of embeddings
        """
        return len(self.encoding_table)

    def __getitem__(self, id):
        """ Retrieve word's embedding. Return the embedding for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.id2emb(id)

    def __setitem__(self, key, value):
        # Raise error, if one tries to edit the VocabEntry.
        raise ValueError('embeddings are readonly')

    def __contains__(self, id):
        """ Check if word is captured by VocabEntry.
        @param id (str): id to look up
        @returns contains (bool): whether id is contained
        """
        return id in range(len(self.encoding_table))

    def add(self, id):
        """ Add id to embeddings, if it is previously unseen.
        @param id (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        new_embed = np.random.normal(scale=0.6, size=(self.emb_dim,))
        if id > len(self.encoding_table):
            self.encoding_table = np.append(self.encoding_table, [new_embed])
        else:
            self.encoding_table[id] = new_embed
