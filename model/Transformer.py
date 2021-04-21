import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, padding_idx, **transformer_config):

        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx)
        ### positional encoding
        self.model = nn.Transformer(d_model = d_model, **transformer_config)

    def forward(self, src_ids, tgt_ids):

        src = self.embed(src_ids)
        tgt = self.embed(tgt_ids)
        tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        logit = self.model(
            src = src,
            tgt = tgt,
            tgt_mask = tgt_mask,
        )
        logit = logit @ self.embed.weight.data.transpose(0, 1)

        return logit

    def incremental_decode(self, src_ids, start_ids, max_len):

        src = self.embed(src_ids)
        tgt = self.embed(start_ids)
        memory = self.model.encoder(src)
        tgt_full_mask = self.model.generate_square_subsequent_mask(max_len).to(src.device)
        output_ids = start_ids
        for i in range(max_len):
            tgt_mask = tgt_full_mask[:i+1, :i+1]
            output = self.model.decoder(tgt, memory, tgt_mask = tgt_mask)
            step_out_ids = torch.max(output[-1], -1)[1].unsqueeze(0)
            step_out_emb = self.embed(step_out_ids)
            output_ids = torch.cat((output_ids, step_out_ids))
            tgt = torch.cat((tgt, step_out_emb))

        return output_ids
        
