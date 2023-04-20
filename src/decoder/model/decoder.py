"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, embedding_sizes, d_model, ffn_hidden, n_head, n_layers, drop_prob, bayes_compression, device):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob,
                                                  bayes_compression=bayes_compression)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, 12)

        self.tok_pred0 = nn.Linear(d_model, embedding_sizes[0])
        self.tok_pred1 = nn.Linear(d_model, embedding_sizes[1])
        self.tok_pred2 = nn.Linear(d_model, embedding_sizes[2])
        self.tok_pred3 = nn.Linear(d_model, embedding_sizes[3])
        self.tok_pred4 = nn.Linear(d_model, embedding_sizes[4])
        self.tok_pred5 = nn.Linear(d_model, embedding_sizes[5])
        self.tok_pred6 = nn.Linear(d_model, embedding_sizes[6])
        self.tok_pred7 = nn.Linear(d_model, embedding_sizes[7])

    def forward(self, tgt, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            tgt = layer(tgt, enc_src, trg_mask, src_mask)

        # Pass hidden state to each head
        out0 = self.tok_pred0(tgt)
        out1 = self.tok_pred1(tgt)
        out2 = self.tok_pred2(tgt)
        out3 = self.tok_pred3(tgt)
        out4 = self.tok_pred4(tgt)
        out5 = self.tok_pred5(tgt)
        out6 = self.tok_pred6(tgt)
        out7 = self.tok_pred7(tgt)

        return out0, out1, out2, out3, out4, out5, out6, out7

    def _kl_divergence(self):
        KLD = 0
        for layer in self.layers:
            KLD += layer._kl_divergence()
        return KLD
