"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class OctupleEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, embedding_sizes, d_model, max_len, drop_prob, PAD_IDX, device):
        """
        class for word embedding that included positional information

        :param embedding_sizes: list, size of vocabulary for every
        :param d_model: dimensions of model
        """
        super(OctupleEmbedding, self).__init__()

        d_embed = int(d_model / 8)
        self.tok_emb0 = TokenEmbedding(embedding_sizes[0], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb1 = TokenEmbedding(embedding_sizes[1], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb2 = TokenEmbedding(embedding_sizes[2], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb3 = TokenEmbedding(embedding_sizes[3], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb4 = TokenEmbedding(embedding_sizes[4], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb5 = TokenEmbedding(embedding_sizes[5], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb6 = TokenEmbedding(embedding_sizes[6], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb7 = TokenEmbedding(embedding_sizes[7], d_embed, PAD_IDX=PAD_IDX)

        self.pos_emb = PositionalEncoding(d_model, drop_prob, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x0_emb = self.tok_emb0(x[:, :, 0])
        x1_emb = self.tok_emb1(x[:, :, 1])
        x2_emb = self.tok_emb2(x[:, :, 2])
        x3_emb = self.tok_emb3(x[:, :, 3])
        x4_emb = self.tok_emb4(x[:, :, 4])
        x5_emb = self.tok_emb5(x[:, :, 5])
        x6_emb = self.tok_emb6(x[:, :, 6])
        x7_emb = self.tok_emb7(x[:, :, 7])

        # TODO LINEAR LAYER

        emb_cat = torch.cat((x0_emb, x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, x7_emb), dim=2)

        pos_emb = self.pos_emb(emb_cat)
        return self.drop_out(emb_cat + pos_emb)
