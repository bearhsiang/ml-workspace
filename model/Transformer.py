import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, padding_idx, **transformer_config):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx)
        ### positional encoding
        self.pos_embed = PositionalEncoding(d_model)
        self.model = nn.Transformer(d_model = d_model, **transformer_config)

        self.output_projection = nn.Linear(
                self.embed.weight.shape[1],
                self.embed.weight.shape[0],
                bias=False,
            )
        self.output_projection.weight = self.embed.weight

        self.apply(self.init_weights)

    def forward(self, src_ids, tgt_ids):

        src = self.embed(src_ids)
        src = self.pos_embed(src)
        tgt = self.embed(tgt_ids)
        tgt = self.pos_embed(tgt)
        tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        logit = self.model(
            src = src,
            tgt = tgt,
            tgt_mask = tgt_mask,
        )
        logit = self.output_projection(logit)

        return logit

    def incremental_decode(self, src_ids, start_ids, max_len):

        src = self.embed(src_ids)
        src = self.pos_embed(src)
        memory = self.model.encoder(src)
        tgt_full_mask = self.model.generate_square_subsequent_mask(max_len).to(src.device)
        output_ids = start_ids
        for i in range(max_len):
            tgt_mask = tgt_full_mask[:i+1, :i+1]
            decoder_out = self.model.decoder(self.pos_emb(tgt), memory, tgt_mask = tgt_mask)
            logit = decoder_out @ self.embed.weight.data.transpose(0, 1)
            step_out_ids = torch.max(output[-1], -1)[1].unsqueeze(0)
            step_out_emb = self.embed(step_out_ids)
            output_ids = torch.cat((output_ids, step_out_ids))
            tgt = torch.cat((tgt, step_out_emb))

        return output_ids
        
    def init_weights(self, module):
        # self.embed.weight.data.normal_(0, 0.02)
        # self.model.encoder.weight.data.uniform_(-initrange, initrange)
        # self.model.decoder.bias.data.zero_()
        # self.model.decoder.weight.data.uniform_(-initrange, initrange)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # if isinstance(module, nn.MultiheadAttention):
        #     module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        #     module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        #     module.v_proj.weight.data.normal_(mean=0.0, std=0.02)