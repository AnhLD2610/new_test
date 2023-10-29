#!/usr/bin/env python
"""
File: tiny_dataset
Date: 3/4/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

from torch.utils.data import Dataset


class TinyDataset(Dataset):

    def __init__(self, paths=None, group=None, num_chunks=None, num_examples=None):

        self.examples = [
            ("this is a tiny data set of just 4 examples . it is for testing and debugging and hopefully it'll help .",
             "this is a tiny test data set ."),

            ("i went to the store and got some flowers .",
             "i got some flowers from the store ."),

            ("turtles are my favorite animal . they are slow and cute and that's why i like them a lot .",
             "i like turtles ."),

            ("i hope that this data set helps us debug the problems with our model . that would be great !",
             "i hope this data set helps .")
        ]

        if num_examples is not None:
            self.examples = self.examples[:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        return self.examples[index]
