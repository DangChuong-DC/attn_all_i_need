from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MHASubLayer
from .feedforward import FFNSubLayer
from .embedding import Embedding
from .positional_encoding import PositionEncoding


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.mha = MHASubLayer(
            n_head=n_head, d_model=d_model, attn_dropout_p=dropout_p, out_dropout_p=dropout_p
        )

        self.ffn = FFNSubLayer(
            d_model=d_model, d_ff=d_ff, dropout_p=dropout_p
        )

    def forward(
        self, src_x: torch.Tensor, src_pad_mask: torch.Tensor.bool | None = None
    ) -> torch.Tensor:
        x, _ = self.mha(src_x, src_x, src_x, src_pad_mask)
        x = self.ffn(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.mmha = MHASubLayer(
            n_head=n_head, d_model=d_model, attn_dropout_p=dropout_p, out_dropout_p=dropout_p
        )

        self.mha = MHASubLayer(
            n_head=n_head, d_model=d_model, attn_dropout_p=dropout_p, out_dropout_p=dropout_p
        )

        self.ffn = FFNSubLayer(
            d_model=d_model, d_ff=d_ff, dropout_p=dropout_p
        )

    def forward(
        self, 
        src_x: torch.Tensor, 
        tgt_x: torch.Tensor,
        merged_tgtpad_attn_mask: torch.Tensor.bool | None = None,
    ) -> torch.Tensor:
        x, _ = self.mmha(tgt_x, tgt_x, tgt_x, merged_tgtpad_attn_mask)
        x, _ = self.mha(x, src_x, src_x)
        x = self.ffn(x)
        return x
    

class Transformer(nn.Module):
    def __init__(
        self,
        voc_size: int,
        max_seq_len: int,
        n_layer: int = 6,
        n_head: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout_p: float = 0.0,
    ) -> None:
        super(Transformer, self).__init__()
        # embedding layer
        self.emb_layer = Embedding(voc_size, d_model=d_model)
        # positional encoding layer
        self.pos_enc = PositionEncoding(max_seq_len, d_model=d_model, dropout_p=dropout_p)

        self.encoder = nn.ModuleList([
            EncoderBlock(
                n_head=n_head, d_model=d_model, d_ff=d_ff, dropout_p=dropout_p
            ) for _ in range(n_layer)
        ])

        self.decoder = nn.ModuleList([
            DecoderBlock(
                n_head=n_head, d_model=d_model, d_ff=d_ff, dropout_p=dropout_p
            ) for _ in range(n_layer)
        ])

        self.linear = nn.Linear(d_model, voc_size)
        # self.softmax = nn.LogSoftmax(dim=-1) # remove as use cross-entropy loss

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_pad_mask: torch.Tensor.bool | None = None,
        merged_mask: torch.Tensor.bool | None = None,
    ) -> torch.Tensor:
        # make embeddings for source text
        src_x = self.emb_layer(src_ids)
        src_x = self.pos_enc(src_x)
        # make embedings for target text
        tgt_x = self.emb_layer(tgt_ids)
        tgt_x = self.pos_enc(tgt_x)

        # process source embedding using encoder layers
        for lay in self.encoder:
            src_x = lay(src_x, src_pad_mask)
        
        # processing using decoder layers
        for lay in self.decoder:
            tgt_x = lay(src_x, tgt_x, merged_mask)
        
        x = self.linear(tgt_x)
        # x = self.softmax(x) # remove as use cross-entropy loss
        return x
