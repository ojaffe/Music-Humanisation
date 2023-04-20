from torch import nn
import torch.nn.functional as F
from models.layers.BayesianLayers import LinearGroupNJ


class CrossAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, dropout:float=0.0, training: bool=False, bayes_compression: bool=False):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        if bayes_compression:
            self.w_q = LinearGroupNJ(embed_dimension, embed_dimension, cuda=True)
            self.w_k = LinearGroupNJ(embed_dimension, embed_dimension, cuda=True)
            self.w_v = LinearGroupNJ(embed_dimension, embed_dimension, cuda=True)
            self.c_proj = LinearGroupNJ(embed_dimension, embed_dimension, cuda=True)

            self.kl_list = [self.w_q, self.w_k, self.w_v, self.c_proj]
        else:
            self.w_q = nn.Linear(embed_dimension, embed_dimension, bias=bias)
            self.w_k = nn.Linear(embed_dimension, embed_dimension, bias=bias)
            self.w_v = nn.Linear(embed_dimension, embed_dimension, bias=bias)
            self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        self.training = training

    def forward(self, q, k, v):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        batch_size = q.size(0)
        embed_dim = q.size(2)
        head_dim = embed_dim // self.num_heads

        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))

        return y

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD


class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0, training: bool=False, bayes_compression: bool=False):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        if bayes_compression:
            self.c_attn = LinearGroupNJ(embed_dimension, 3 * embed_dimension, cuda=True)
            # output projection
            self.c_proj = LinearGroupNJ(embed_dimension, embed_dimension, cuda=True)
            self.kl_list = [self.c_attn, self.c_proj]
        else:
            self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
            # output projection
            self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal
        self.training = training

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y

    def _kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD
