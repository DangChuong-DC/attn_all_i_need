import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        d_model: int = 512,
        dropout_p: float = 0.0,
    ) -> None:
        super(PositionEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.dropout_p = dropout_p

        position = torch.arange(0, max_seq_len, 1, dtype=torch.float).unsqueeze(1) # (S, 1)

        div_term = torch.exp(-torch.arange(0, d_model, 2,  dtype=torch.float) * math.log(10000) / d_model).unsqueeze(0) # (1, D)

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # (S, D)
        pe = pe.unsqueeze(0) # adding batch dimension
        self.register_buffer("positional_encoding", pe)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        output = embeddings + self.positional_encoding
        if self.dropout_p > 0.0:
            output = F.dropout(output, p=self.dropout_p, training=self.training)
        return output


if __name__ == "__main__":
    frequency = torch.pow(10000, -torch.arange(0, 18, 2, dtype=torch.float)/128)
    div_term = torch.exp(-torch.arange(0, 18, 2, dtype=torch.float) * math.log(10000) / 128)

    print(frequency)
    print(div_term)

    div_term = div_term.unsqueeze(0)
    position = torch.arange(0, 3, 1, dtype=torch.float).unsqueeze(1)

    test_out = position * div_term
    print(test_out.size())
    print(test_out)
