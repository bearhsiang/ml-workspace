from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):

    def __init__(self, prefix, langs, suffix=''):
        super().__init__()
        self.data = {}
        self.length = -1
        self.langs = langs
        for lang in langs:
            self.data[lang] = []
            with open(prefix+lang+suffix, 'r') as f:
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
    dataset = PairedDataset("data/wmt17_en-de/train/raw.", ['en', 'de'])
    for lang in dataset.data:
        print(len(dataset.data[lang]))
    # for i in range(10):
    #     print(dataset[i])
    # testset = DataLoader(dataset, batch_size=10, shuffle=True)
    # for batch in testset:
    #     print(batch)
    #     break