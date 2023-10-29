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

class DatasetGroup(Enum):
    train = 1
    validation = 2
    test = 3