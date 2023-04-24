"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from transformer.blocks.decoder_layer import DecoderLayer
from transformer.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, num_heads, n_layers, drop_prob, octuple, octuple_em_sizes, flash, PAD_IDX, device):
        super().__init__()
        self.octuple = octuple

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  num_heads=num_heads,
                                                  drop_prob=drop_prob,
                                                  flash=flash)
                                     for _ in range(n_layers)])

        if not self.octuple:
            self.emb = TransformerEmbedding(d_model=d_model,
                                            drop_prob=drop_prob,
                                            max_len=max_len,
                                            vocab_size=dec_voc_size,
                                            PAD_IDX=PAD_IDX,
                                            device=device)
            self.linear = nn.Linear(d_model, dec_voc_size)
        else:
            self.tok_pred0 = nn.Linear(d_model, octuple_em_sizes[0])
            self.tok_pred1 = nn.Linear(d_model, octuple_em_sizes[1])
            self.tok_pred2 = nn.Linear(d_model, octuple_em_sizes[2])
            self.tok_pred3 = nn.Linear(d_model, octuple_em_sizes[3])
            self.tok_pred4 = nn.Linear(d_model, octuple_em_sizes[4])
            self.tok_pred5 = nn.Linear(d_model, octuple_em_sizes[5])
            self.tok_pred6 = nn.Linear(d_model, octuple_em_sizes[6])
            self.tok_pred7 = nn.Linear(d_model, octuple_em_sizes[7])


    def forward(self, enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask, s_mask):
        if not self.octuple:
            dec = self.emb(dec)

        for layer in self.layers:
            dec = layer(enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask, s_mask)

        # pass to LM head
        if self.octuple:
            out0 = self.tok_pred0(dec)
            out1 = self.tok_pred1(dec)
            out2 = self.tok_pred2(dec)
            out3 = self.tok_pred3(dec)
            out4 = self.tok_pred4(dec)
            out5 = self.tok_pred5(dec)
            out6 = self.tok_pred6(dec)
            out7 = self.tok_pred7(dec)

            return out0, out1, out2, out3, out4, out5, out6, out7
        else:
            output = self.linear(dec)
            return output
