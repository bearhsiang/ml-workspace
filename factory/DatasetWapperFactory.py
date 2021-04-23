from datasetWapper import PairedDataset

def get_dataset(name, split, config):
    if name == "PairedDataset":
        return PairedDataset(split, config)
