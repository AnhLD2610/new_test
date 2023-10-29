#!/usr/bin/env python
"""
File: dataset_types
Date: 2/24/19
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

from enum import Enum
from config import config


class DatasetType(Enum):
    """ each of the differnet data set types that can be used """
    tiny = 0
    cnn = 1
    nyt = 2
    giga = 3


# just a mapping from data set name to type
dataset_names = {
    "cnn": DatasetType.cnn,
    "nyt": DatasetType.nyt,
    "giga" : DatasetType.giga,
    "tiny": DatasetType.tiny
}


def get_dataset_class(dataset_type):
    """ Returns the classes which represent a specified data set.
    The data set may be specified either by name (a string) or
    with the Enum of data set. Note, we do the importing down here so that
    we don't end up importing things that aren't used.

    :param dataset_type: A string specifying the name of the data set to use OR a DatasetType Enum value
    :param small: Whether to use the small version of the data set of not
    :return: An object which represents the data set layout on disk (i.e. paths), and a class
    derived from torch.data.Dataset for getting examples from that data set.
    """

    if type(dataset_type) is str:
        if dataset_type not in dataset_names:
            raise ValueError("No such data set named \"%s\"" % dataset_type)
        return get_dataset_class(dataset_names[dataset_type])

    if dataset_type == DatasetType.tiny:
        # we return the CNNDailyMailPaths so that we can steal it's Vocab file
        from CNNDailyMail import CNNDailyMailPaths
        from data.tiny_dataset import TinyDataset
        return CNNDailyMailPaths(config.cnn_dataset_path), TinyDataset
    if dataset_type == DatasetType.cnn:
        from CNNDailyMail import CNNDMDataset, CNNDailyMailPaths
        return CNNDailyMailPaths(config.cnn_dataset_path), CNNDMDataset
    if dataset_type == DatasetType.nyt:
        from NYTDataset import NYTDataset, NYTPaths
        return NYTPaths(config.nyt_dataset_path, small=False), NYTDataset
    if dataset_type == DatasetType.giga:
        from NYTDataset import NYTDataset, NYTPaths
        return NYTPaths(config.nyt_dataset_path, small=True), NYTDataset

    raise ValueError("No such data set: %s" % dataset_type)
