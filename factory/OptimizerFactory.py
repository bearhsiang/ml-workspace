import torch.optim as optim
from optimizer import NoamOpt

def get_optimizer(param, type, config={}):
    if type == 'NoamOpt':
        backend_optimizer = get_optimizer(param, **config['backend'])
        return NoamOpt(backend_optimizer, **config)
    if type == 'AdamW':
        return optim.AdamW(param, **config)