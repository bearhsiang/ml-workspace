from tokenizers import SentencePieceBPETokenizer, trainers, models, Tokenizer
from pathlib import Path

vocab_size = 800

data_root = Path('./data')
dataset_name = 'wmt17_en-de'
# splits = ['train', 'validation']
splits = ['train']
langs = ['en', 'de']
output_path = str(data_root/dataset_name/f'vocab_{vocab_size}.json')

training_files = []
for split in splits:
    for lang in langs:
        training_files.append(str(data_root / dataset_name / split / f'raw.{lang}'))

tokenizer = Tokenizer(models.WordPiece())
trainer = trainers.WordPieceTrainer(vocab_size=vocab_size)
tokenizer.train(training_files, trainer=trainer)
tokenizer.save(output_path)

