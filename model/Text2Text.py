from torch.nn import nn
from .Seq2Seq import Seq2Seq
from factory import get_model

class Text2Text(Seq2Seq):

    def __init__(self, in_vocab_size, out_vocab_size, share_embed, add_pos_embed=False, **Seq2SeqConfig):

        self.seq2seq_model = get_model(**Seq2SeqConfig)
        self.in_embed = nn.Embedding(in_vocab_size, self.seq2seq_model.get_encoder_dim)
        self.add_pos_embed = add_pos_embed

        if self.add_pos_embed:
            self.pos_embed = PositionalEncoding(self.seq2seq_model.get_encoder_dim)

        if not share_emb:
            self.out_embed = nn.Embedding(out_vocab_size, self.seq2seq_model.get_decoder_dim())
        else:
            self.out_embed = self.in_embed

    def forward(self, src_ids, prev_out_ids):

        enc_input = self.in_embed(src_ids)
        dec_input = self.out_embed(prev_out_ids)

        if self.add_pos_embed:
            enc_input = self.pos_embed(enc_input)
            dec_input = self.pos_embed(dec_input)

        logit = self.seq2seq_model(enc_input, dec_input)
        
        logit = logit @ self.out_embed.weight.data.T

        return logit

