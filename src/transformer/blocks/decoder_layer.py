"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from transformer.layers.layer_norm import LayerNorm
from transformer.layers.position_wise_feed_forward import PositionwiseFeedForward
from transformer.layers.fast_attention import CausalSelfAttention, CrossAttention


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, flash):
        super(DecoderLayer, self).__init__()
        #self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.num_heads = num_heads
        self.flash = flash

        self.fast_self_attention = CausalSelfAttention(num_heads=num_heads,
                                                       embed_dimension=d_model,
                                                       bias=False,
                                                       is_causal=True,
                                                       training=True)
        self.fast_cross_attention = CrossAttention(num_heads=num_heads,
                                                   embed_dimension=d_model,
                                                   bias=False,
                                                   training=True)

        self.enc_dec_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask, s_mask):
        # 1. compute self attention
        _x = dec

        if self.flash:
            # Construct combined pad and causual mask
            if enc_pad_mask is None:
                flash_mask = None
            else:
                batch_size, seq_len = dec_pad_mask.size()
                n_heads = int(dec_causal_mask.shape[0] / batch_size)

                dec_pad_mask_reshaped = dec_pad_mask.unsqueeze(1).repeat(1, seq_len, 1)
                dec_causal_mask_reshaped = dec_causal_mask[0].unsqueeze(0).repeat(batch_size, 1, 1) 

                combined_mask = torch.bitwise_or(dec_pad_mask_reshaped, dec_causal_mask_reshaped)  # (b, s, s)
                combined_mask_stretched = combined_mask.unsqueeze(1).repeat((1, n_heads, 1, 1))  # (b, h, s, s)
                flash_mask = torch.ne(combined_mask_stretched, True)  # True where attention should be applied

            x = self.fast_self_attention(x=dec, flash_mask=flash_mask)
        else:
            # Line 1243 https://github.com/pytorch/pytorch/blob/ee5f09ab802a04cc966829230c236e9132041ab7/torch/nn/modules/activation.py
            # stupid pytorch code
            if not self.training:
                dec_causal_mask = dec_causal_mask[0]

            x, _ = self.self_attention(dec, dec, dec, key_padding_mask=dec_pad_mask.bool(), attn_mask=dec_causal_mask.bool())
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            
            if self.flash:
                if enc_pad_mask is None:
                    flash_mask = None
                else:
                    enc_pad_mask_reshaped = enc_pad_mask.unsqueeze(1).unsqueeze(1)
                    enc_pad_mask_reshaped = enc_pad_mask_reshaped.repeat(1, self.num_heads, x.shape[1], 1)
                    flash_mask = torch.ne(enc_pad_mask_reshaped, True)  # True where attention should be applied

                x = self.fast_cross_attention(q=x, k=enc, v=enc, flash_mask=flash_mask)
            else:
                x, _ = self.enc_dec_attention(x, enc, enc)
                
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
