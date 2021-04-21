import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from dataset import PairedDataset
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from tqdm.auto import tqdm
import datasets

def create_batch(text_batch, tokenizer, max_len=-1):

    batch = {}
    key_list = []
    bin_batch_list = []
    
    for key in text_batch:
        tensor_list = [ torch.tensor(i) for i in tokenizer.encode(text_batch[key])]
        bin_batch = nn.utils.rnn.pad_sequence(tensor_list, padding_value=tokenizer.pad_id())
        if max_len > 0:
            batch[key] = bin_batch[:max_len]
        else:
            batch[key] = bin_batch

    return batch

def train_one_epoch(model, optimizer, data, tokenizer, criterion, config, valid_data=None):
    
    model.train()
    optimizer.zero_grad()

    device = next(model.parameters()).device

    bar = tqdm(data)

    for step, text_batch in enumerate(bar):
        batch = create_batch(text_batch, tokenizer, config['max_len'])
        src_ids = batch[config['src_lang']].to(device)
        tgt_ids = batch[config['tgt_lang']].to(device)
        logit = model(src_ids, tgt_ids)
        shifted_tgt_ids = torch.cat((tgt_ids[1:], torch.full_like(tgt_ids[:1], tokenizer.pad_id())), 0)
        loss = criterion(logit.view(-1, logit.size(-1)), shifted_tgt_ids.view(-1))
        loss.backward()
        if (step+1) % config['grad_accu_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        bar.set_postfix({
            'loss': loss.item(),
        })
        if (step+1) % config['valid_step'] == 0 and valid_data:
            score = inference(model, valid_data, tokenizer, config)
            print(score)

def inference(model, data, tokenizer, config, incremental_decode=False):
    metric = datasets.load_metric('sacrebleu')
    model.eval()
    bar = tqdm(data)
    device = next(model.parameters()).device
    with torch.no_grad():
        for text_batch in bar:
            batch = create_batch(text_batch, tokenizer, config['valid_max_len'])
            src_ids = batch[config['src_lang']].to(device)
            if incremental_decode:
                start_ids = torch.full((1, src_ids.size(1)), tokenizer.bos_id()).to(device)
                hyp_ids = model.inference(src_ids, start_ids, max_len=64)
            else:
                tgt_ids = batch[config['tgt_lang']].to(device)
                logit = model(src_ids, tgt_ids)
                hyp_ids = torch.max(logit, -1)[1]
            src_text = text_batch[config['src_lang']]
            hyp_text = tokenizer.decode(hyp_ids.T.tolist())
            ref_text = [[s] for s in text_batch[config['tgt_lang']]]
            metric.add_batch(predictions=hyp_text, references=ref_text)

        for i in range(5):
            print('src:', src_text[i])
            print('hyp:', hyp_text[i])
            print('ref:', ref_text[i])

    score = metric.compute()
    return score


def main():

    config = {
        'src_lang': 'de',
        'tgt_lang': 'en',
        'total_epochs': 30,
        'spm_model_path': 'data/wmt17_en-de/spm_8000.model',
        'train_batch_size': 64,
        'grad_accu_step': 16,
        'valid_step': 1024,
        'valid_batch_size': 128,
        'max_len': 128,
        'valid_max_len': 128,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = spm.SentencePieceProcessor(model_file=config['spm_model_path'])
    tokenizer.set_encode_extra_options("bos:eos")
    train_data = DataLoader(
        PairedDataset('data/wmt17_en-de/train/raw.', [config['src_lang'], config['tgt_lang']]),
        batch_size = config['train_batch_size'],
        shuffle = True,
    )
    valid_data = DataLoader(
        PairedDataset('data/wmt17_en-de/validation/raw.', [config['src_lang'], config['tgt_lang']]),
        batch_size = config['valid_batch_size'],
        shuffle = False,
    )

    model_config = {
        'vocab_size': tokenizer.vocab_size(),
        'd_model': 128,
        'padding_idx': tokenizer.pad_id(),
    }
    model = Transformer(**model_config).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    for epoch in range(config['total_epochs']):
        train_one_epoch(model, optimizer, train_data, tokenizer, criterion, config, valid_data=valid_data)
        score = inference(model, valid_data, tokenizer, config)
        print(score)

        # batch = create_batch(text_batch, tokenizer)
        # for key in batch:
        #     print(batch[key].shape)
        #     print(tokenizer.decode(batch[key].T.tolist()))
        #     print(text_batch[key])
        # break


if __name__ == '__main__':
    main()