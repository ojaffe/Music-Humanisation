import utils

import math as m
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn


def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = utils.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0, training: bool=False):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
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

    def forward(self, x, mask):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        #query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        #key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        #value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048, flash=False):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model

        self.flash = flash
        if flash:
            self.mha = CausalSelfAttention(num_heads=h, embed_dimension=d_model, bias=False, is_causal=False, dropout=rate)
        else:
            #self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)
            self.rga = torch.nn.MultiheadAttention(embed_dim=self.d_model, num_heads=h, batch_first=True)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, **kwargs):
        if self.flash:
            attn_out = self.mha(x, mask)
            w = None
        else:
            #attn_out, w = self.rga([x, x, x], mask)
            attn_out, w = self.rga(x, x, x, key_padding_mask=mask, is_causal=True)
        
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, h, input_vocab_size, rate=0.1, max_len=None, flash=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, rate, h=h, additional=False, max_seq=max_len, flash=flash)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask)
            weights.append(w)
        return x, weights # (batch_size, input_seq_len, d_model)


# class MusicTransformerDataParallelCriterion(torch.nn.DataParallel):
#     def forward(self, inputs, *targets, **kwargs):
#         targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
#         replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
#         targets = tuple(targets_per_gpu[0] for targets_per_gpu in targets)
#         outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
#         return Reduce.apply(*outputs) / len(outputs), targets
