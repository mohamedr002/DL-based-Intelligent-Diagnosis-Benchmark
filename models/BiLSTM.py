import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __init__(self, batch_size=64, in_channel=1, out_channel=10):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 64
        self.num_layers = 3
        V = 128
        C = out_channel
        self.dropout = nn.Dropout(0.5)
        self.embed = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(V))
        self.hidden2label1 = nn.Linear(self.hidden_dim * 2 * V, self.hidden_dim * 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim * 2, C)
        self.hidden = self.init_hidden(self.num_layers, batch_size)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)))

    def forward(self, x): # BS*1*-1
        x = self.embed(x) # BS*16*V
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)  # V*BS*16
        x = self.dropout(x)
        bilstm_out, self.hidden = self.bilstm(x, self.hidden) # V*BS*2H
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2) # BS*2H*V
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1) # BS*2HV
        logit = self.hidden2label1(bilstm_out) # BS*2H
        logit = self.hidden2label2(logit) # BS*C

        return logit
