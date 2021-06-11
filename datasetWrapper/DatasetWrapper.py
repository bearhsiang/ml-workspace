from torch.utils.data import Dataset
class DatasetWrapper(Dataset):
    
    def __init__(self, **config):
        super().__init__()
        pass

    def load_data(self, dataset, split):
        pass

    def __getitem__(self):
        pass

    def __len__(self):
        pass