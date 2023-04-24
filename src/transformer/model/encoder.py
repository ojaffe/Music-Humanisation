"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from transformer.blocks.encoder_layer import EncoderLayer
from transformer.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, num_heads, n_layers, drop_prob, octuple, flash, PAD_IDX, device):
        super().__init__()
        self.octuple = octuple

        if not octuple:
            self.emb = TransformerEmbedding(d_model=d_model,
                                            max_len=max_len,
                                            vocab_size=enc_voc_size,
                                            drop_prob=drop_prob,
                                            PAD_IDX=PAD_IDX,
                                            device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  num_heads=num_heads,
                                                  drop_prob=drop_prob,
                                                  flash=flash)
                                     for _ in range(n_layers)])

    def forward(self, enc, enc_pad_mask):
        if not self.octuple:
            enc = self.emb(enc)

        for layer in self.layers:
            enc = layer(enc, enc_pad_mask)

        return enc
