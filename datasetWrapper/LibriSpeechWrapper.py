from torch.utils.data import Dataset, DataLoader
import os
import torchaudio

class LibriSpeechWrapper(Dataset):

    SAMPLE_RATE = 16000

    def __init__(self, **config):
        super().__init__()

    def load_data(self, data_dir, prefix, **other):

        prefix = os.path.join(data_dir, prefix)
        self.data_dir = prefix
        self.data = []

        for speaker_id in os.listdir(prefix):
            if not speaker_id.isnumeric():
                continue
            for chapter_id in os.listdir(os.path.join(prefix, speaker_id)):
                if not chapter_id.isnumeric():
                    continue
                trans_file_name = f'{speaker_id}-{chapter_id}.trans.txt'
                for line in open(os.path.join(prefix, speaker_id, chapter_id, trans_file_name), 'r'):
                    line = line.strip().split(' ', 1)
                    self.data.append(line)


    def __getitem__(self, idx):
        tag = self.data[idx][0]
        text = self.data[idx][1]
        speaker_id, chapter_id, sentence_id = tag.split('-')
        wav_path = os.path.join(self.data_dir, speaker_id, chapter_id, f'{tag}.flac')
        wav = self._load_wav(wav_path)
        return wav, text

    def _load_wav(self, path):
        wav, sr = torchaudio.load(path)
        assert sr == self.SAMPLE_RATE
        return wav

    def collate_fn(self, samples):
        
        wavs, texts = [], []
        for wav, text in samples:
            wavs.append(wav)
            texts.append(text)

        return wavs, texts

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    root_dir = '/hdd/LibriSpeech'
    split = 'test-clean'

    libri = LibriSpeechWapper()
    libri.load_data(root_dir, split)
    dataloader = DataLoader(libri, batch_size=10, shuffle=True, collate_fn=getattr(libri, 'collate_fn', None))
    for batch in dataloader:
        print(batch)
        break