# many-to-many seq2seq problem: varying number of frames to (varying/ different) number of words.
import torch
import torch.nn as nn


class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LstmEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = None
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        # lstm_out.shape    : (N, L, hidden_size)
        # self.hidden.shape : (1, N, hidden_size)
        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = None
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.linear = nn.Linear(hidden_size, input_size)  # can be replaced with proj size in LSTM

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
        # lstm_out.shape    : (N, 1, hidden_size)
        output = self.linear(self.dropout(lstm_out.squeeze(1)))

        # output.shape : (N, 1, input_size)
        # self.hidden.shape : (1, N, hidden_size)
        return output, self.hidden
