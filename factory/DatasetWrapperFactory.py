from datasetWrapper import PairedDatasetWrapper

def get_datasetWrapper(type, config={}):
    if type == "PairedDatasetWrapper":
        return PairedDatasetWrapper(**config)
