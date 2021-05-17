import torch.nn as nn

class AttentionalRNNDecoder(nn.Module):

    def __init__(self):

        self.rnn1 = nn.LSTM()
        self.rnn2 = nn.LSTM()

        self.multi_attn = nn.MultiheadAttention()

    def forward(self, features, prev_out_features):
        
        state1 = None
        state2 = None
        
        output = []

        for i in range(prev_out_features.size(0)):
            out1, state2 = self.rnn1(prev_out_features[i].unsqueeze(0), state1)
            attn_out = self.multi_attn(out1, features, features)[0]
            out2, state1 = self.rnn2(attn_out, state2)
            output.append(out2)

        


            

