import copy
from .base_model import BaseModel

__all__ = ['build_model']

def build_model(config):
    config = copy.deepcopy(config)
    return BaseModel(config)
