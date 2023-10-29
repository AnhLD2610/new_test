#!/usr/bin/env python
"""
File: Configuration
Date: 2/9/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os
import configparser


class Configuration(object):
    """
    A class simply for holding configurations (i.e. paths to data files, etc.)
    """

    def __init__(self, config_file):
        # Setup the filesystem configuration
        self._config_file = os.path.join(config_file)
        self._config = configparser.ConfigParser()
        self._config.read(self._config_file)
        c = self._config

        self.cnn_dataset_path = os.path.expanduser(c["Data"]["cnn_path"])
        self.nyt_dataset_path = os.path.expanduser(c["Data"]["nyt_path"])
        self.glove_emb_path = os.path.expanduser(c["Data"]["emb_path"])

        self.use_cuda = bool(c["Cuda"]["use"])

    def override(self, settings):
        """
        Allows for variables in the configuration to be over-ridden.
        All attributes of "settings" which are also attributes
        of this object will be set to the values found in "settings"
        :param settings: Object with attributes to override in this object
        :return: None
        """
        for attr in vars(settings):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                value = getattr(settings, attr)
                setattr(self, attr, value)
