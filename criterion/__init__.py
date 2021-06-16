import torch
from .LabelSmoothingLoss import LabelSmoothingLoss

def get_criterion(type, config, tokenizer=None):

    if type.split('.')[0] == 'torch':
        if type == 'torch.nn.CrossEntropyLoss':
            config['ignore_index'] = tokenizer.pad_id()
            return eval(type)(**config)
    
    if type == 'LabelSmoothingLoss':
        config['classes'] = tokenizer.vocab_size()
        config['ignore_index'] = tokenizer.pad_id()
        return LabelSmoothingLoss(**config)

    raise NotImplementedError