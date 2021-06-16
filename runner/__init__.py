from .TranslationRunner import *
from .ASRRunner import *

def get_runner(type, config):
    if type == 'TranslationRunner':
        return TranslationRunner(**config)
    elif type == 'ASRRunner':
        return ASRRunner(**config)