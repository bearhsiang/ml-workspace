from model import Transformer

def get_model(type, config):
    if type == 'Transformer':
        return Transformer(**config)