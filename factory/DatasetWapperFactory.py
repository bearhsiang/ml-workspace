from datasetWapper import PairedDatasetWapper

def get_datasetWapper(type):
    if type == "PairedDatasetWapper":
        return PairedDatasetWapper()
