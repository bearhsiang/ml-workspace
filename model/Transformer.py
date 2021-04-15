import torch
import torch.nn as nn

class Transformer(nn.module):

    def __init__(self, config):
        super().__init__():
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.model = nn.Transformer(**config)

    def forward(self,input_ids):

        emb = self.embed(input_ids)
        logit = self.model(emb)

        return logit