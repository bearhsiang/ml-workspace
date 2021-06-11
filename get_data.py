from datasets import load_dataset
from pathlib import Path
from tqdm.auto import tqdm

dataset_name = 'wmt17_en-cs'
dataset_config = ['wmt17', 'cs-en']
langs = ['en', 'cs']

data_root = Path('./data')
dataset = load_dataset(*dataset_config)

for split in dataset:
    lang_fs = {}
    data_dir = data_root / dataset_name / split
    data_dir.mkdir(parents=True)
    for lang in langs:
        lang_fs[lang] = open(data_dir / f'raw.{lang}', 'w')
    for data in tqdm(dataset[split]['translation']):
        for lang in langs:
            print(data[lang].replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').strip(), file = lang_fs[lang])
