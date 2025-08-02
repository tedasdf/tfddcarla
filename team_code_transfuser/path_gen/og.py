import torch.nn as nn


class GRUDecoder(nn.Module):
    def __init__(self, gru_hidden_size, gru_concat_target_point, device):
        super().__init__()
        self.gru_concat_target_point = gru_concat_target_point
        self.gru_cell = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2, # 2 represents x,y coordinate
                                  hidden_size=gru_hidden_size).to(device)
        self.linear = nn.Linear(gru_hidden_size, 3)

    def forward(self, x, h):
        h = self.gru_cell(x, h)
        out = self.linear(h)
        return out, h
