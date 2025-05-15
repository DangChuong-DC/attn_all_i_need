import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
            self,
            voc_size: int,
            d_model: int = 512,
        ) -> None:
        super(Embedding, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model

        self.emb_layer = nn.Embedding(voc_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.emb_layer(input_ids)
        embeddings = embeddings * math.sqrt(self.d_model)
        return embeddings
    

if __name__ == "__main__":
    emb_layer = Embedding(128, d_model=128)

    in_ids = torch.randint(low=0, high=128, size=(1, 13))
    # print(in_ids)
    test_out = emb_layer(in_ids)
    print(test_out.shape)
    print(test_out.dtype)
    print(test_out)
