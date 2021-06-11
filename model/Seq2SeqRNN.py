import torch.nn as nn
import torch

class Seq2SeqRNN(nn.Module):

    def __init__(self, vocab_size, padding_idx, d_model, n_layers, bidirectional=True):

        super().__init__()

        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.n_layers = n_layers

        if bidirectional:
            assert d_model % 2 == 0
            d_encoder = d_model // 2
        else:
            d_encoder = d_model

        self.encoder = nn.LSTM(
            input_size = d_model,
            hidden_size = d_encoder,
            num_layers = n_layers,
            bidirectional = bidirectional,
        )

        self.decoder = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
        )

    def forward(self, src_ids, tgt_ids):
        
        src_emb = self.embedding(src_ids)

        _, state = self.encoder(src_emb)

        state = self.transform_state(state)

        decoder_input_emb = self.embedding(tgt_ids)
        logit, _ = self.decoder(decoder_input_emb, state)
        predict = logit @ self.embedding.weight.transpose(0, 1)
        
        return predict

    def transform_state(self, state):

        h, c = state

        h = h.view(self.n_layers, -1, h.size(1), h.size(2))
        h = torch.cat([h[:, i, :, :] for i in range(h.size(1))], dim=-1)

        c = c.view(self.n_layers, -1, c.size(1), c.size(2))
        c = torch.cat([c[:, i, :, :] for i in range(c.size(1))], dim=-1)

        return (h, c)

    def inference(self, src_ids, start_ids, max_len):

        features = self.embedding(src_ids)
        _, state = self.encoder(features)
        state = self.transform_state(state)
        output_ids = start_ids
        for i in range(max_len):
            decoder_input_emb = self.embedding(output_ids[-1]).unsqueeze(0)
            logit, state = self.decoder(decoder_input_emb, state)
            predict = logit @ self.embedding.weight.transpose(0, 1)
            predict_id = torch.argmax(predict, dim=-1)
            output_ids = torch.cat((output_ids, predict_id), dim=0)
        return output_ids