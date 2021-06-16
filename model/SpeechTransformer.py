import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
import logging

logger = logging.getLogger(__name__)

class SpeechTransformer(nn.Module):

    def __init__(self, vocab_size, d_model, padding_idx, connector, share_decoder_input_output_embed=False, **transformer_config):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.padding_idx = padding_idx

        from . import get_model
        ### connector
        self.connector = get_model(**connector)

        ### positional encoding
        self.pos_embed = PositionalEncoding(d_model)
        self.model = nn.Transformer(d_model = d_model, **transformer_config)
        self.d_model = d_model
        
        self.output_projection = nn.Linear(
                self.embed.weight.shape[1],
                self.embed.weight.shape[0],
                bias=False,
        )
        if share_decoder_input_output_embed:
            self.output_projection.weight = self.embed.weight
        
        self.apply(self.init_weights)

    def forward(self, src_features, src_lengths, tgt_ids):

        src, src_lengths = self.connector(src_features, src_lengths)
        src = self.pos_embed(src)

        tgt = self.embed(tgt_ids)
        tgt = self.pos_embed(tgt)

        src_key_padding_mask = self._create_mask_with_lengths(src_lengths, src.size(0))
        tgt_key_padding_mask = (tgt_ids == self.padding_idx).transpose(0, 1)
        tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        logit = self.model(
            src = src,
            tgt = tgt,
            src_mask = None, 
            tgt_mask = tgt_mask,
            memory_mask = None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logit = self.output_projection(logit)

        return logit

    def inference(self, src_features, src_lengths, start_ids, max_len):

        src, src_lengths = self.connector(src_features, src_lengths)
        src = self.pos_embed(src)
        src_key_padding_mask = self._create_mask_with_lengths(src_lengths, src.size(0))

        memory = self.model.encoder(
            src, src_key_padding_mask=src_key_padding_mask, 
        )
        
        output_ids = start_ids

        for i in range(max_len):

            tgt_mask = self.model.generate_square_subsequent_mask(output_ids.size(0)).to(src.device)
            tgt = self.embed(output_ids)
            tgt = self.pos_embed(tgt)

            tgt_key_padding_mask = (output_ids == self.padding_idx).transpose(0, 1).to(src.device)

            logit = self.model.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask = None,
                tgt_key_padding_mask = tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

            logit = self.output_projection(logit)
            step_out_ids = torch.argmax(logit[-1], dim=-1).unsqueeze(0)
            output_ids = torch.cat((output_ids, step_out_ids), dim=0)

        return output_ids
        
    def init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.d_model**0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _create_mask_with_lengths(self, lengths, max_lengths=-1):

        batch_size = lengths.size(0)
        if max_lengths < 0:
            max_lengths = torch.max(lengths).item()
        mask = torch.arange(max_lengths).to(lengths.device).view(1, max_lengths)
        mask = mask.expand(batch_size, -1) >= lengths.view(batch_size, 1).expand(-1, max_lengths)
        return mask
