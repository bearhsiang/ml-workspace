from .Runner import Runner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer
import sacrebleu
import logging

logger = logging.getLogger(__name__)

class TranslationRunner(Runner):

    def __init__(self, src_lang, tgt_lang, max_len, tokenizer, **other_config):
        super().__init__(**other_config)

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # max_len is an dict(), with key 'train' and 'valid'
        self.max_len = max_len

        self.tokenizer = get_tokenizer(**tokenizer)
        self.tokenizer.set_encode_extra_options("bos:eos")

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
        tgt_ids = tgt_ids[:self.max_len[mode]]

        if mode == 'train':

            tgt = tgt_ids[1:]

            model_input = {
                'src_ids': src_ids,
                'tgt_ids': tgt_ids[:-1],
            }

            return model_input, tgt

        ## infernece
        model_input = {
            'src_ids': src_ids,
            'start_ids': tgt_ids,
            'max_len' : min(self.max_len[mode], int(src_ids.size(0) * 1.2 + 10)),
        }

        return model_input

    def set_model(self, **config):
        config['config']['vocab_size'] = self.tokenizer.vocab_size()
        config['config']['padding_idx'] = self.tokenizer.pad_id()
        super().set_model(
            **config,
        )
    
    def set_criterion(self, **config):
        self.criterion = super().set_criterion(**config, tokenizer=self.tokenizer).to(self.device)

    def count_loss(self, hypo, gold):
        hypo = hypo.reshape(-1, hypo.size(-1))
        gold = gold.reshape(-1)
        return self.criterion(hypo, gold)
        
    def step(self, raw_batch, log, mode):
        if mode == 'train':
            super().step(raw_batch, log, mode)
        elif mode == 'valid':
            batch = self.create_batch(raw_batch, mode)
            out_ids = self.model.inference(**batch).transpose(0, 1).cpu().tolist()

            # batch = self.create_batch(raw_batch, mode='train')
            # model_input, tgt = batch
            # logit = self.model(**model_input)
            # out_ids = torch.argmax(logit, dim=-1).transpose(0, 1).cpu().tolist()

            # remove tokens behind <eos>
            for i in range(len(out_ids)):
                if self.tokenizer.eos_id() in out_ids[i]:
                    out_ids[i] = out_ids[i][:out_ids[i].index(self.tokenizer.eos_id())]

            hyp_text = self.tokenizer.decode(out_ids)

            log['src'] += raw_batch[self.src_lang]
            # log['src_tokens'] += [' '.join([self.tokenizer.id2token(i) for i in s]) for s in model_input['src_ids'].transpose(0, 1).cpu().tolist()]
            log['hyp'] += hyp_text
            # log['hyp_tokens'] += [' '.join([self.tokenizer.id2token(i) for i in s]) for s in out_ids]
            log['ref'] += raw_batch[self.tgt_lang]
            # log['ref_tokens'] += [' '.join([self.tokenizer.id2token(i) for i in s]) for s in model_input['tgt_ids'].transpose(0, 1).cpu().tolist()]
            # log['tgt_tokens'] += [' '.join([self.tokenizer.id2token(i) for i in s]) for s in tgt.transpose(0, 1).cpu().tolist()]

    def log(self, log, mode, split):

        if mode == 'train':

            loss = sum(log['loss'])
            self.monitor.log({
                'loss': loss,
            })

        elif mode == 'valid':

            bleu = sacrebleu.corpus_bleu(log['hyp'], [log['ref']])
            self.monitor.log({
                f'{split}-bleu': bleu.score,  
            })
            logger.info(f'{split}-bleu: {bleu.score}')
            for i in range(10):
                logger.info(f'[src] {log["src"][i]}')
                # logger.info(f'[src tokens] {log["src_tokens"][i]}')
                logger.info(f'[ref] {log["ref"][i]}')
                # logger.info(f'[ref tokens] {log["ref_tokens"][i]}')
                # logger.info(f'[tgt tokens] {log["tgt_tokens"][i]}')
                logger.info(f'[hyp] {log["hyp"][i]}')
                # logger.info(f'[hyp tokens] {log["hyp_tokens"][i]}')


# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()

#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))

#         self.confidence = 1.0 - label_smoothing

#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

#         return F.kl_div(output, model_prob, reduction='batchmean')