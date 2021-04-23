import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from dataset import PairedDataset
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from tqdm.auto import tqdm
import datasets
from optimizer.NoamOpt import NoamOpt
import logging
import sacrebleu

logging.basicConfig(format='|%(levelname)s| %(message)s', level=logging.DEBUG)

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
    
    optimizer.zero_grad()

    device = next(model.parameters()).device

    bar = tqdm(data)

    for step, text_batch in enumerate(bar):
        model.train()
        batch = create_batch(text_batch, tokenizer, config['max_len'])
        enc_in_ids = batch[config['src_lang']].to(device)
        dec_in_ids = batch[config['tgt_lang']].to(device)
        logit = model(enc_in_ids, dec_in_ids)
        tgt_ids = torch.cat((dec_in_ids[1:], torch.full_like(dec_in_ids[:1], tokenizer.pad_id())), 0)
        loss = criterion(logit.view(-1, logit.size(-1)), tgt_ids.view(-1))
        bar.set_postfix({
            'loss': loss.item(),
        })
        loss.backward()
        if (step+1) % config['grad_accu_step'] == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step+1) % config['valid_step'] == 0 and valid_data:
            score = inference(model, valid_data, tokenizer, config)


def inference(model, data, tokenizer, config, incremental_decode=False):

    model.eval()
    bar = tqdm(data)
    device = next(model.parameters()).device
    srcs = []
    hyps = []
    refs = []
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
            ref_text = text_batch[config['tgt_lang']]

            hyp_ids = hyp_ids.T.tolist()
            clean_hyp_ids = []
            for l in hyp_ids:
                if tokenizer.eos_id() in l:
                    clean_hyp_ids.append(l[:l.index(tokenizer.eos_id())])
                else:
                    clean_hyp_ids.append(l)
            hyp_text = tokenizer.decode(clean_hyp_ids)


            srcs += src_text
            refs += ref_text
            hyps += hyp_text
            

        for i in range(5):
            print('src:', src_text[i])
            print('hyp:', hyp_text[i])
            print('ref:', ref_text[i])

    tok = 'zh' if config['tgt_lang'] == 'zh' else '13a'
    score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    logging.info(f'bleu score = {score.score}')
    logging.info(f'precision = {score.precisions}')
    return score


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    config = {
        'src_lang': 'de',
        'tgt_lang': 'en',
        'total_epochs': 30,
        'spm_model_path': 'data/wmt17_en-de/spm_8000.model',
        'train_prefix': 'data/wmt17_en-de/train/clean.',
        'valid_prefix': 'data/wmt17_en-de/validation/raw.',
        'train_batch_size': 64,
        'grad_accu_step': 2,
        'valid_step': 1024,
        'valid_batch_size': 128,
        'max_len': 128,
        'valid_max_len': 128,
        'lr_factor': 2,
        'lr_warmup': 4000,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = spm.SentencePieceProcessor(model_file=config['spm_model_path'])
    tokenizer.set_encode_extra_options("bos:eos")
    train_data = DataLoader(
        PairedDataset(config['train_prefix'], [config['src_lang'], config['tgt_lang']]),
        batch_size = config['train_batch_size'],
        shuffle = True,
    )
    valid_data = DataLoader(
        PairedDataset(config['valid_prefix'], [config['src_lang'], config['tgt_lang']]),
        batch_size = config['valid_batch_size'],
        shuffle = False,
    )

    model_config = {
        'vocab_size': tokenizer.vocab_size(),
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'padding_idx': tokenizer.pad_id(),
    }
    model = Transformer(**model_config).to(device)
    optimizer = NoamOpt(
        model_size=model_config['d_model'], 
        factor=config['lr_factor'], 
        warmup=config['lr_warmup'], 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    for epoch in range(config['total_epochs']):
        logging.info(f'epoch {epoch}:')
        train_one_epoch(model, optimizer, train_data, tokenizer, criterion, config, valid_data=valid_data)
        score = inference(model, valid_data, tokenizer, config)

        # batch = create_batch(text_batch, tokenizer)
        # for key in batch:
        #     print(batch[key].shape)
        #     print(tokenizer.decode(batch[key].T.tolist()))
        #     print(text_batch[key])
        # break


if __name__ == '__main__':
    main()
