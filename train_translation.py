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

@hydra.main(config_path="conf", config_name="translation_en-de.yaml")
def main(config : DictConfig):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(**config['tokenizer'])
    train_set = get_datasetWapper(**config['datasetWapper'])
    train_set.load_data(config['dataset'], 'train')
    valid_set = get_datasetWapper(**config['datasetWapper'])
    valid_set.load_data(config['dataset'], 'valid')
    # tokenizer.set_encode_extra_options("bos:eos")
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

    config['model']['config']['vocab_size'] = tokenizer.vocab_size()
    config['model']['config']['padding_idx'] = tokenizer.pad_id()
    model = get_model(**config['model']).to(device)
    optimizer = get_optimizer(model.parameters(), **config['optimizer'])

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
