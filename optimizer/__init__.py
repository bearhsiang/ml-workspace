from .NoamOpt import *

import torch

def get_optimizer(param, type, config={}):
    if type == 'NoamOpt':
        backend_optimizer = get_optimizer(param, **config['backend'])
        return NoamOpt(backend_optimizer, **config)

    if type.split('.')[0] == 'torch':
        return eval(type)(param, **config)

    raise NotImplementedError