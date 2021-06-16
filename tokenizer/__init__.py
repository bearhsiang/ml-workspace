from .SPTokenizer import *

def get_tokenizer(type, config):
    if type == 'SPTokenizer':
        return SPTokenizer(**config)