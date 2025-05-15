from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        dropout_p: float = 0.0,
    ) -> None:
        super(SelfAttention, self).__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        qry: torch.Tensor, 
        key: torch.Tensor, 
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert qry.size(-1) == key.size(-1), "query and key feature must have same feature dimension `d_k`"

        attn_bias = None
        if attn_mask is not None:
            assert (
                attn_mask.ndim == 3
            ), f"`attn_mask` must have dimension as (B, S_q, S_k) or (B, 1, S_k), got {attn_mask.size()}"
            B = attn_mask.size(0) # batch size
            M, N = qry.size(-2), key.size(-2)
            assert (
                (attn_mask.size(-2) == M or attn_mask.size(-2) == 1) and attn_mask.size(-1) == N
            ), "size mismatch between mask and features"
            attn_bias = torch.zeros(B, M, N, dtype=qry.dtype, device=qry.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), -math.inf)
            if qry.ndim == 4:
                attn_bias = attn_bias.unsqueeze(1)
            attn_bias = attn_bias.to(dtype=qry.dtype)
        
        scaled_dot_prod = torch.matmul(qry, key.transpose(-2, -1)) / math.sqrt(qry.size(-1))
        if attn_bias is not None:
            scaled_dot_prod += attn_bias

        attn_weights = F.softmax(scaled_dot_prod, dim=-1)
        # if attn_mask is not None:
        #     attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).logical_not(), 0)
        if self.dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        output = torch.matmul(attn_weights, val)
        return output, attn_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        attn_dropout_p: float = 0.0,
        out_dropout_p: float = 0.0,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "model dimension must dividable by num of attention head!"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.attention = SelfAttention(dropout_p=attn_dropout_p)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = out_dropout_p
    
    def forward(
        self,
        qry: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W_q(qry)
        k = self.W_k(key)
        v = self.W_v(val)

        # splitting into number of head
        q = q.view(q.size(0), q.size(1), self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_head, self.head_dim).transpose(1, 2)

        attn_feats, attn_weights = self.attention(q, k, v, attn_mask)
        # concatinate attention feature
        attn_feats = attn_feats.transpose(1, 2).reshape(qry.size(0), qry.size(1), self.d_model)

        output = self.W_o(attn_feats)
        if self.dropout_p > 0.0:
            output = F.dropout(output, p=self.dropout_p, training=self.training)
        return output, attn_weights
    

class MHASubLayer(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        attn_dropout_p: float = 0.0,
        out_dropout_p: float = 0.0,
    ) -> None:
        super(MHASubLayer, self).__init__()
        self.layer = MultiHeadAttention(
            n_head=n_head, d_model=d_model, attn_dropout_p=attn_dropout_p, out_dropout_p=out_dropout_p
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        qry: torch.Tensor,
        key: torch.Tensor,
        val: torch.Tensor,
        attn_mask: torch.Tensor.bool | None = None
    ):
        res, attn_wei = self.layer(qry, key, val, attn_mask)
        output = self.norm(qry + res)
        return output, attn_wei
        

if __name__ == "__main__":
    # pad_mask = torch.ones((1, 9))
    # for i in range(6, 9):
    #     pad_mask[:, i] = 0

    # pad_mask = pad_mask.unsqueeze(-1)
    # pad_mask = torch.matmul(pad_mask, pad_mask.transpose(-2, -1))
    # pad_mask = pad_mask.to(dtype=torch.bool)
    # print(pad_mask)

    # attn_bias = torch.randn((4, 1, 13, 13))
    # dot_prod = torch.randn((4, 8, 13, 13))
    # test_out = dot_prod + attn_bias
    # print(test_out.shape)
    # print(test_out.ndim)

    mha = MultiHeadAttention(n_head=8, d_model=128, attn_dropout_p=0.1, out_dropout_p=0.1)

    qry = torch.randn((4, 13, 128))
    key = torch.randn((4, 13, 128))
    val = torch.randn((4, 13, 128))
    test_out, test_w = mha(qry, key, val)

    print(test_out.shape)
    print(test_w.size())
    print(test_w[:, :, 0, :].sum(dim=-1))
