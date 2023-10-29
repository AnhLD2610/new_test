#!/usr/bin/env python
"""
File: __init__.py
Date: 2/9/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

import os
from config.Configuration import Configuration

dir_name = os.path.dirname(__file__)
default_config_file = os.path.join(dir_name, "config.ini")

# import this variable to import the global configuration
config = Configuration(default_config_file)