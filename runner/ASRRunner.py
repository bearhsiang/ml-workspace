from .Runner import Runner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer
import sacrebleu
import logging
import torchaudio
import editdistance

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
            tgt_ids = [[self.tokenizer.bos_id()] for _ in range(len(src_features))]

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

        wer = []

        for hyp, ref in zip(hypo, gold):
            hyp_w = hyp.split()
            ref_w = ref.split()
            wer.append(editdistance.eval(hyp_w, ref_w)/len(ref_w))

        wer = sum(wer)/len(wer)

        return wer

    def step(self, raw_batch, log, mode):
        if mode == 'train':
            super().step(raw_batch, log, mode)
        elif mode == 'valid':
            batch = self.create_batch(raw_batch, mode)
            out_ids = self.model.inference(**batch).transpose(0, 1).cpu().tolist()

            # remove tokens behind <eos>
            for i in range(len(out_ids)):
                if self.tokenizer.eos_id() in out_ids[i]:
                    out_ids[i] = out_ids[i][:out_ids[i].index(self.tokenizer.eos_id())]

            hyp_text = self.tokenizer.decode(out_ids)

            log['hyp'] += hyp_text
            log['ref'] += raw_batch[1]

    def log(self, log, mode, split):

        if mode == 'train':

            loss = sum(log['loss'])
            self.monitor.log({
                f'{mode}-{split}/loss': loss,
            })

        elif mode == 'valid':

            score = self._metric(log['hyp'], log['ref'])
            self.monitor.log({
                f'{mode}-{split}/wer': score,
            })
            for i in range(5):
                logger.info(f"{mode}-{split}/{i}")
                logger.info(f"{mode}-{split}/[hyp] {log['hyp'][i]}")
                logger.info(f"{mode}-{split}/[ref] {log['ref'][i]}")
            logger.info(f'{mode}-{split}/wer: {score}')

