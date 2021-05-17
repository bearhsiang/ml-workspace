import torch.nn as nn
from factory import get_model

class Seq2Seq(nn.Module):

    def __init__(self, encoder_config, decoder_config):

        self.encoder = get_model(**encoder_config)
        self.decoder = get_model(**decoder_config)

    def forward(self, in_features, prev_out_features, momery=None):

        if memory is None:
            memroy = self.encoder(in_features)

        return self.decoder(momory, prev_out_features)

    def get_encoder_dim(self):
        pass

    def get_decoder_dim(self):
        pass