"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch

from transformer.embedding.positional_encoding import PositionalEncoding
from transformer.embedding.token_embeddings import TokenEmbedding


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

        # Decrease dimensionality of program, since always piano
        program_d_embed = 4
        pitch_d_embed = ((3 * d_embed) - program_d_embed) // 2
        dur_d_embed = ((3 * d_embed) - program_d_embed) - pitch_d_embed

        d_embed = int(d_model / 8)
        self.tok_emb0 = TokenEmbedding(embedding_sizes[0], program_d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb1 = TokenEmbedding(embedding_sizes[1], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb2 = TokenEmbedding(embedding_sizes[2], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb3 = TokenEmbedding(embedding_sizes[3], pitch_d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb4 = TokenEmbedding(embedding_sizes[4], dur_d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb5 = TokenEmbedding(embedding_sizes[5], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb6 = TokenEmbedding(embedding_sizes[6], d_embed, PAD_IDX=PAD_IDX)
        self.tok_emb7 = TokenEmbedding(embedding_sizes[7], d_embed, PAD_IDX=PAD_IDX)

        self.fc = nn.Linear(d_model, d_model)

        self.pos_emb = PositionalEncoding(d_model, max_len, device)
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

        emb_cat = torch.cat((x0_emb, x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, x7_emb), dim=2)

        lin = self.fc(emb_cat)
        
        pos_emb = self.pos_emb(x[:, :, 0])
        return self.drop_out(lin + pos_emb)
