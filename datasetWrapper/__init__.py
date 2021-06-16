from .PairedDatasetWrapper import *
from .DatasetWrapper import *
from .LibriSpeechWrapper import * 

def get_dataset(wrapper, config):
    wrapper = get_datasetWrapper(**wrapper)
    wrapper.load_data(**config)
    return wrapper

def get_datasetWrapper(type, config={}):
    if type == "PairedDatasetWrapper":
        return PairedDatasetWrapper(**config)
    if type == "LibriSpeechWrapper":
        return LibriSpeechWrapper(**config)

    raise NotImplementedError
