"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from transformer.layers.layer_norm import LayerNorm
from transformer.layers.position_wise_feed_forward import PositionwiseFeedForward
from transformer.layers.fast_attention import CausalSelfAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, flash):
        super(EncoderLayer, self).__init__()
        #self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.num_heads = num_heads
        self.flash = flash

        self.fast_self_attention = CausalSelfAttention(num_heads=num_heads,
                                                       embed_dimension=d_model,
                                                       bias=False,
                                                       is_causal=False,
                                                       training=True)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, enc, enc_pad_mask):
        # 1. compute self attention
        _enc = enc

        if self.flash:
            enc_pad_mask_reshaped = enc_pad_mask.unsqueeze(1).unsqueeze(1)
            enc_pad_mask_reshaped = enc_pad_mask_reshaped.repeat(1, self.num_heads, enc_pad_mask.shape[1], 1)
            enc_pad_mask_reshaped_ne = torch.ne(enc_pad_mask_reshaped, True)  # True where attention should be applied

            enc = self.fast_self_attention(x=enc, flash_mask=enc_pad_mask_reshaped_ne)
        else:
            enc, _ = self.self_attention(enc, enc, enc, key_padding_mask=enc_pad_mask)
        
        # 2. add and norm
        enc = self.dropout1(enc)
        enc = self.norm1(enc + _enc)
        
        # 3. positionwise feed forward network
        _enc = enc
        enc = self.ffn(enc)
      
        # 4. add and norm
        enc = self.dropout2(enc)
        enc = self.norm2(enc + _enc)
        return enc
