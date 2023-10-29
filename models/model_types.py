#!/usr/bin/env python
"""
File: model_types
Date: 2/9/19 
Author(s):
    - Jon Deaton (jdeaton@stanford.edu)
    - Austin Jacobs (ajacobs7@stanford.edu)
    - Kathleen Kenealy (kkenealy@stanford.edu)
"""

from enum import Enum


class ModelType(Enum):
    transformer = 1
    pointer = 2


model_names = {
    "baseline": ModelType.transformer,
    "pointer": ModelType.pointer
}


def get_model_class(model_type):
    """ Get the class that represents the model type.
    Note: we import the model types down here so that we don't end up
    importing things that aren't used.

    :param model_type: string name OR ModelType Enum of the model type
    :return: Class representing the model type
    """
    if type(model_type) is str:
        if model_type not in model_names:
            raise ValueError("No such model type: %s" % model_type)
        return get_model_class(model_names[model_type])

    if model_type == ModelType.pointer:
        from models.transformer_pointer import TransformerPointer
        return TransformerPointer

    if model_type == ModelType.transformer:
        from models.transformer_baseline import Baseline
        return Baseline

    raise ValueError("No such model type: %s" % model_type)
