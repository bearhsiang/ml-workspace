from tokenizer import SPTokenizer

def get_tokenizer(type, config):
    if type == 'SPTokenizer':
        return SPTokenizer(**config)