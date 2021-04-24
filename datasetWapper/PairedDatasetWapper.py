from torch.utils.data import Dataset, DataLoader
from .DatasetWapper import DatasetWapper

class PairedDatasetWapper(DatasetWapper):

    def __init__(self):
        super().__init__()

    def load_data(self, dataset, split):
        self.langs = dataset['keys']
        self.data = {}
        self.length = -1
        prefix = f"{dataset['data_dir']}/{dataset[split]['prefix']}"
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
        'train':{
            'prefix': 'train/clean',
        },
        'valid':{
            'prefix': 'validation/raw',
        },
    }
    dataset = PairedDatasetWapper()
    dataset.load_data(dataset_config, 'valid')
    for i in range(10):
        print(dataset[i])
    # for i in range(10):
    #     print(dataset[i])
    # testset = DataLoader(dataset, batch_size=10, shuffle=True)
    # for batch in testset:
    #     print(batch)
    #     break