from .Trainer import Trainer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from factory import get_tokenizer
import sacrebleu

class TranslationTrainer(Trainer):

    def __init__(self, src_lang, tgt_lang, max_len, tokenizer, **other_config):

        super().__init__(**other_config)

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # max_len is an dict(), with key 'train' and 'valid'
        self.max_len = max_len

        self.tokenizer = get_tokenizer(**tokenizer)

    def create_batch(self, raw_batch, mode='valid'):

        src_text = raw_batch[self.src_lang]
        src_ids = self.tokenizer.encode(src_text)
        
        if mode == 'train':
            tgt_text = raw_batch[self.tgt_lang]
            tgt_ids = self.tokenizer.encode(tgt_text)
        else:
            tgt_ids = [[self.tokenizer.bos_id()] for _ in range(len(src_ids))]

        src_ids = [torch.LongTensor(l) for l in src_ids]
        tgt_ids = [torch.LongTensor(l) for l in tgt_ids]

        src_ids = pad_sequence(
            src_ids, 
            batch_first = False,
            padding_value = self.tokenizer.pad_id(),
        ).to(self.device)

        tgt_ids = pad_sequence(
            tgt_ids,
            batch_first = False, 
            padding_value = self.tokenizer.pad_id(),
        ).to(self.device)

        src_ids = src_ids[:self.max_len[mode]]
        tgt_ids = src_ids[:self.max_len[mode]]

        if mode == 'train':

            tgt = torch.cat([
                tgt_ids[1:], 
                torch.full((1, tgt_ids.size(1)),
                    fill_value=self.tokenizer.pad_id()).to(self.device)
                ], dim=0)

            model_input = {
                'src_ids': src_ids,
                'tgt_ids': tgt_ids,
            }

            return model_input, tgt

        ## infernece
        model_input = {
            'src_ids': src_ids,
            'start_ids': tgt_ids,
            'max_len' : self.max_len,
        }

        return model_input

    def set_model(self, **config):
        config['config']['vocab_size'] = self.tokenizer.vocab_size()
        config['config']['padding_idx'] = self.tokenizer.pad_id()
        super().set_model(
            **config
        )

    def set_criterion(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id())

    def count_loss(self, hypo, gold):
        hypo = hypo.view(-1, hypo.size(-1))
        gold = gold.view(-1)
        return self.criterion(hypo, gold)
        
    def step(self, raw_batch, log, mode):

        if mode == 'train':
            super().step(raw_batch, log, mode)
        elif mode == 'valid':
            batch = self.create_batch(raw_batch, mode)
            out = self.model.inference(**batch)
            out_ids = torch.argmax(out, dim=-1)
            hyp_text = self.tokenizer.decode(out_ids.transpose(0, 1).cpu())
            log['hyp'] += hyp_text
            log['ref'] += raw_batch[self.tgt_lang]

    def log(self, log, mode, split):

        if mode == 'train':

            loss = sum(log['loss']) / len(log['loss'])
            self.monitor.log({
                'loss': loss,
            })

        elif mode == 'valid':

            bleu = sacrebleu.corpus_bleu(log['hyp'], [log['ref']])
            self.monitor.log({
                f'{split}-bleu': bleu.score,  
            })

            
            

    
