"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from tqdm import tqdm

import torch
from torch import nn

from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder
from transformer.embedding.octuple_embedding import OctupleEmbedding


class Transformer(nn.Module):

    def __init__(self, cfg, octuple, vocab_size, max_example_len, SOS_IDX, EOS_IDX, PAD_IDX, device):
        super().__init__()
        self.octuple = octuple
        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX
        self.device = device

        self.encoder = Encoder(enc_voc_size=vocab_size, 
                               max_len=max_example_len, 
                               d_model=cfg.get("d_model"), 
                               ffn_hidden=cfg.get("ffn_hidden"), 
                               num_heads=cfg.get("num_heads"), 
                               n_layers=cfg.get("num_layers"), 
                               drop_prob=cfg.get("dropout"), 
                               octuple=octuple,
                               flash=cfg.get("flash"),
                               PAD_IDX=PAD_IDX,
                               device=device)

        self.decoder = Decoder(d_model=cfg.get("d_model"),
                               max_len=max_example_len,
                               dec_voc_size=vocab_size,
                               num_heads=cfg.get("num_heads"),
                               ffn_hidden=cfg.get("ffn_hidden"),
                               drop_prob=cfg.get("dropout"),
                               n_layers=cfg.get("num_layers"),
                               octuple=octuple,
                               octuple_em_sizes=cfg["octuple_em_sizes"],
                               flash=cfg.get("flash"),
                               PAD_IDX=PAD_IDX,
                               device=device)

        self.oct_emb = OctupleEmbedding(embedding_sizes=cfg["octuple_em_sizes"],
                                    d_model=cfg.get("d_model"),
                                    max_len=max_example_len,
                                    drop_prob=cfg.get("dropout"),
                                    PAD_IDX=PAD_IDX,
                                    device=device)

    def forward(self, enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask):
        if self.octuple:
            enc = self.oct_emb(enc)
            dec = self.oct_emb(dec)

        #enc = self.encoder(enc, enc_pad_mask)
        enc = None

        output = self.decoder(enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask, None)
        return output

    def generate(self, src, tgt_in, length):
        decoded_tokens = tgt_in

        input_tokens = torch.tensor([tgt_in]).to(self.device)
        with torch.no_grad():
            for i in tqdm(range(length)):
                logits = self.forward(src, input_tokens, None)

                logits = logits[:, logits.shape[1]-1, :]
                top_token = torch.argmax(logits).item()

                if top_token == self.EOS_IDX:
                    break

                decoded_tokens.append(top_token)
                input_tokens = torch.cat((input_tokens, torch.tensor([[top_token]]).to(self.device)), dim=1)

        return decoded_tokens
    
    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()

    def set_test(self):
        self.eval()
