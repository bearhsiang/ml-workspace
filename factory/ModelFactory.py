from model import *

def get_model(type, config):
    if type == 'Transformer':
        return Transformer(**config)
    elif type == 'Seq2SeqRNN':
        return Seq2SeqRNN(**config)
    elif type == 'SpeechTransformer':
        return SpeechTransformer(**config)
    else:
        raise NotImplementedError