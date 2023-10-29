#!/usr/bin/env python
"""
File: __init__.py
Date: 2/11/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)

Slightly modified from:
https://github.com/abisee/pointer-generator/blob/master/data.py
"""

from data.vocab import article2ids, abstract2ids

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

UNK_ID = 0
PAD_ID = 1
START_ID = 2
STOP_ID = 3


PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

