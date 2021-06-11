from datasetWrapper import *

def get_datasetWrapper(type, config={}):
    if type == "PairedDatasetWrapper":
        return PairedDatasetWrapper(**config)
    if type == "LibriSpeechWrapper":
        return LibriSpeechWrapper(**config)

    raise NotImplementedError
