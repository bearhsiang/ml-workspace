from .Runner import Runner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer
import sacrebleu
import logging
import torchaudio

logger = logging.getLogger(__name__)

class ASRRunner(Runner):

    def __init__(self, max_len, tokenizer, **other_config):
        super().__init__(**other_config)
        
        # max_len is an dict(), with key 'train' and 'valid'
        self.max_len = max_len

        self.tokenizer = get_tokenizer(**tokenizer)
        self.tokenizer.set_encode_extra_options("bos:eos")

    def create_batch(self, raw_batch, mode='valid'):

        src_wavs = raw_batch[0]
        src_features = [torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80) for wav in src_wavs]
        if self.max_len[mode]['feature'] > 0:
            src_features = [feature[: self.max_len[mode]['feature']] for feature in src_features]
        
        src_lengths = torch.LongTensor([len(feature) for feature in src_features]).to(self.device)
        
        if mode == 'train':
            tgt_text = raw_batch[1]
            tgt_ids = self.tokenizer.encode(tgt_text)
        else:
            tgt_ids = [[self.tokenizer.bos_id()] for _ in range(len(src_ids))]

        tgt_ids = [torch.LongTensor(l) for l in tgt_ids]

        src_features = pad_sequence(
            src_features, 
            batch_first = False,
            padding_value = self.tokenizer.pad_id(),
        ).to(self.device)

        tgt_ids = pad_sequence(
            tgt_ids,
            batch_first = False, 
            padding_value = self.tokenizer.pad_id(),
        ).to(self.device)

        if self.max_len[mode]['target'] > 0:
            tgt_ids = tgt_ids[:self.max_len[mode]['target']]

        if mode == 'train':

            tgt = tgt_ids[1:]

            model_input = {
                'src_features': src_features,
                'src_lengths': src_lengths,
                'tgt_ids': tgt_ids[:-1],
            }

            return model_input, tgt

        ## infernece
        model_input = {
            'src_features': src_features,
            'src_lengths': src_lengths,
            'start_ids': tgt_ids,
            'max_len' : self.max_len[mode]['target'],
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

    def _metric(self, hypo, gold):

        cer = []
        wer = []

        for hyp, ref in zip(hypo, gold):
            cer.append(editdistance.eval(hyp, ref)/len(ref))
            hyp_w = hyp.split()
            ref_w = ref.split()
            wer.append(editdistance.eval(hyp_w, ref_w)/len(ref_w))

        cer = sum(cer)/len(cer)
        wer = sum(wer)/len(wer)

        return cer, wer

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
