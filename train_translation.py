import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import Transformer
from tqdm.auto import tqdm
import datasets
import logging
import sacrebleu
import hydra
from omegaconf import DictConfig, OmegaConf
from factory import get_tokenizer, get_datasetWapper, get_optimizer, get_model
import os
import wandb
from collections import defaultdict

logger = logging.getLogger(__name__)

def create_batch(text_batch, tokenizer, max_len=-1):

    batch = {}
    key_list = []
    bin_batch_list = []
    
    for key in text_batch:
        tensor_list = [ torch.tensor([tokenizer.bos_id()] + i + [tokenizer.eos_id()]) for i in tokenizer.encode(text_batch[key])]
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

        tgt_ids = dec_in_ids[1:]
        logit = logit[:-1]
        
        # print(f'dec_in_ids: {dec_in_ids[:, 0]}')
        # print(f'tgt_ids: {tgt_ids[:, 0]}')
        # print(f'dec in text: {tokenizer.decode(dec_in_ids[:, 0].cpu().tolist())}')
        # print(f'tgt text: {tokenizer.decode(tgt_ids[:, 0].cpu().tolist())}')
        # print(logit.size())
        # raise

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
            if config['use_wandb']:
                wandb.log({'val bleu':score.score})

        record = {
            'loss': loss.item(),
            'lr': optimizer.rate(),
        }

        if config['use_wandb']:
            wandb.log(record)
            


def inference(model, data, tokenizer, config, incremental_decode=True):

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
            logger.info(f'[src] {srcs[-i]}')
            logger.info(f'[ref] {refs[-i]}')
            logger.info(f'[hyp] {hyps[-i]}')


    tok = 'zh' if config['tgt_lang'] == 'zh' else '13a'
    score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
    logger.info(f'bleu score = {score.score}')
    logger.info(f'precision = {score.precisions}')
    return score

def step(batch, tokenizer, model, log):

    batch = create_batch()

@hydra.main(config_path="conf", config_name="translation.yaml")
def main(config : DictConfig):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load tokenizer
    tokenizer = get_tokenizer(**config['tokenizer'])
    
    # load dataset
    train_set = get_datasetWapper(**config['datasetWapper'])
    train_set.load_data(config['dataset'], 'train')
    valid_set = get_datasetWapper(**config['datasetWapper'])
    valid_set.load_data(config['dataset'], 'valid')
    
    train_data = DataLoader(
        train_set,
        batch_size = config['train_batch_size'],
        shuffle = True,
    )
    
    valid_data = DataLoader(
        valid_set,
        batch_size = config['valid_batch_size'],
        shuffle = False,
    )

    # load model
    config['model']['config']['vocab_size'] = tokenizer.vocab_size()
    config['model']['config']['padding_idx'] = tokenizer.pad_id()
    model = get_model(**config['model'])
    model.to(device)
    
    # load optimizer
    optimizer = get_optimizer(model.parameters(), **config['optimizer'])
    
    # load criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())
    
    # load step
    epoch = 0
    inner_step = 0

    # load checkpoint
    if os.path.exists('checkpoint_last.pt'):
        logger.info('Find checkpoint_last.pt, load from checkpoint')
        checkpoint = torch.load('checkpoint_last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        inner_step = checkpoint['inner_step']

    if config['use_wandb']:
        wandb.init(**config['wandb'], config=config)
        wandb.watch(model)


    bar = tqdm(range(config['total_epochs']), initial=epoch, desc="epoch: ")

    for epoch in bar:

        inner_bar = tdqm(train_data, initial=inner_step//config['train_batch_size'], desc='train: ')

        for batch in inner_bar:

            batch = create_batch(batch, tokenizer, config['max_len'])
            log = defaultdict(list)
            loss = step(model, log, mode=train)
            loss.backward()

            if (step+1) % config['grad_accu_step'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            if (step+1) % config['valid_step'] == 0:

                for split in valid_data:
                    
                    log = defaultdict(list)
                    valid_bar = tqdm(valid_data[split], desc=f'{split}: ')
                    for valid_batch in valid_bar:
                        valid_batch = create_batch(valid_batch, tokenizer, config['max_len'])

                        step(model, log, model=split)
                    
                    score = count_score(log, tokenize)

            if (step+1) % config['log_step'] == 0:


        logger.info(f'echo {epoch} finish, save checkpoints')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, 'checkpoint_last.pt')
        epoch += 1

if __name__ == '__main__':
    main()
