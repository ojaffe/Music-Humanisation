from custom.layers import *
from custom.criterion import *
from custom.layers import Encoder
from custom.config import config

import sys
import torch
import torch.distributions as dist
import random
import utils

import torch
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar


class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, h=8, flash=False, debug=False, loader_path=None, dist=False, writer=None, PAD_IDX=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        """
        decoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model=self.embedding_dim,
                                                         dim_feedforward=2048,
                                                         dropout=dropout,
                                                         batch_first=False)
        self.Decoder = torch.nn.TransformerEncoder(decoder_layer,
                                                   num_layers=num_layer)"""

        self.writer = writer
        self.Decoder = Encoder(
            num_layers=self.num_layer, h=h, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq, flash=flash)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

        self.flash = flash
        self.PAD_IDX = PAD_IDX

    def forward(self, x, length=None, writer=None):
        if self.training or not self.infer:

            pad_mask = utils.build_pad_mask(x, self.PAD_IDX)
            decoder, w = self.Decoder(x, mask=pad_mask)
            fc = self.fc(decoder)
            return fc.contiguous()
            """
            if self.flash:
                mask = utils.build_causal_pad_mask(x, self.PAD_IDX)
                decoder, w = self.Decoder(x, mask=mask)
                fc = self.fc(decoder)
                return fc.contiguous() if self.training else (fc.contiguous(), None)
            else:
                _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(x, x, self.PAD_IDX)

                decoder, w = self.Decoder(x, mask=look_ahead_mask)
                fc = self.fc(decoder)
                return fc.contiguous() if self.training else (fc.contiguous(), [weight.contiguous() for weight in w])"""
        else:
            return self.generate(x, length, None).contiguous().tolist()

    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 tf_board_writer: SummaryWriter = None):
        decode_array = prior
        result_array = prior
        #print(config)
        #print(length)
        for i in range(length):
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array, decode_array, pad_token=self.PAD_IDX)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            result, _ = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = 0
            if u > 1:
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        result_array = result_array[0]
        return result_array

    def set_train(self):
        self.train()
        self.infer = False

    def set_eval(self):
        self.eval()
        self.infer = False

    def set_test(self):
        self.eval()
        self.infer = True
