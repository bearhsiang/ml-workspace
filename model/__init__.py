from .Transformer import *
from .Seq2SeqRNN import *
from .SpeechTransformer import *
from .Conv1dSubsampler import *

def get_model(type, config):
    if type == 'Transformer':
        return Transformer(**config)
    elif type == 'Seq2SeqRNN':
        return Seq2SeqRNN(**config)
    elif type == 'SpeechTransformer':
        return SpeechTransformer(**config)
    elif type == 'Conv1dSubsampler':
        return Conv1dSubsampler(**config)
    else:
        raise NotImplementedError