from torch.utils.data import Dataset, DataLoader
# from .DatasetWapper import DatasetWapper
from torch.utils.data import Dataset

class PairedDatasetWrapper(Dataset):

    def __init__(self, **config):
        super().__init__()

    def load_data(self, data_dir, prefix, keys, **other):

        self.langs = keys
        self.data = {}
        self.length = -1
        prefix = f"{data_dir}/{prefix}"
        for lang in self.langs:
            self.data[lang] = []
            with open(f'{prefix}.{lang}', 'r') as f:
                for line in f:
                    self.data[lang].append(line.strip())
            
            ### make sure that the number of lines are the same

            if self.length < 0:
                self.length = len(self.data[lang])
            else:
                assert self.length == len(self.data[lang])

    def __getitem__(self, index):
        entry = {}
        for lang in self.langs:
            entry[lang] = self.data[lang][index]
        return entry

    def __len__(self):
        return self.length

if __name__ == '__main__':
    dataset_config = {
        'keys': ['en', 'de'],
        'data_dir': './data/wmt17_en-de',
        'prefix': 'validation/raw',
    }
    dataset = PairedDatasetWapper(**dataset_config)
    for i in range(10):
        print(dataset[i])
    # for i in range(10):
    #     print(dataset[i])
    # testset = DataLoader(dataset, batch_size=10, shuffle=True)
    # for batch in testset:
    #     print(batch)
    #     break