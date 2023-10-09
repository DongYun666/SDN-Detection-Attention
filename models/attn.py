import torch
import torch.nn as nn

from math import sqrt
from einops import rearrange
from torch.nn import functional as F



class FullAttention(nn.Module):
    def __init__(self):
        super(FullAttention, self).__init__()

    def forward(self, queries, keys, values):
        B, H, W, L = queries.shape
        _, _, S, D = values.shape
        scale = 1. / sqrt(L)
        scores = torch.einsum("bhle,bhse->bhls", queries.contiguous(), keys.contiguous())
        weight = torch.softmax(scale * scores, dim=-1)
        out = torch.einsum("bhls,bhsd->blhd", weight, values.contiguous())
        return out.contiguous()

class LinearAttentionLayer(nn.Module):
    def __init__(self,attention,d_model,num_heads):
        super(LinearAttentionLayer,self).__init__()
        self.inner_attention = attention
        self.num_heads = num_heads
        self.linear_poj_q = nn.Linear(d_model,d_model*num_heads)
        self.linear_poj_k = nn.Linear(d_model,d_model*num_heads)
        self.linear_poj_v = nn.Linear(d_model,d_model*num_heads)
        self.proj_out = nn.Linear(d_model*num_heads,d_model)

    def forward(self,x):
        b, w, _ = x.shape
        query = rearrange(self.linear_poj_q(x),'b w (h l) -> b h w l',h = self.num_heads)
        key = rearrange(self.linear_poj_k(x),'b w (h l) -> b h w l',h = self.num_heads)
        value = rearrange(self.linear_poj_v(x),'b w (h l) -> b h w l',h = self.num_heads)
        att_out = self.inner_attention(query,key,value) # b h w l
        out = att_out.transpose(1,2).contiguous().view(b,w,-1)
        return self.proj_out(out)

