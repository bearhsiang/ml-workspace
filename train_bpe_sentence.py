import sentencepiece as spm
from pathlib import Path

vocab_size = 8000

data_root = Path('./data')
dataset_name = 'wmt17_en-de'
# splits = ['train', 'validation']
splits = ['train']
langs = ['en', 'de']
output_prefix = str(data_root/dataset_name/f'spm_{vocab_size}')

training_files = []
for split in splits:
    for lang in langs:
        training_files.append(str(data_root / dataset_name / split / f'clean.{lang}'))

spm.SentencePieceTrainer.train(
    input=','.join(training_files),
    model_prefix=output_prefix,
    vocab_size=vocab_size,
    input_sentence_size=1e6*len(langs),
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
)
