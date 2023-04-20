import utils
import random
import pickle
import numpy as np

from custom.config import config


class Data:
    def __init__(self, dir_path, SOS_IDX, EOS_IDX):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX


    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'


    def batch(self, batch_size, length, mode='train'):
        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data)  # batch_size, seq_len


    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1 - 2, mode)  # -2 for SOS and EOS
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y


    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                #raise IndexError
                data = np.append(data, config.token_eos)
                while len(data) < max_length:
                    data = np.append(data, config.pad_token)

        data = np.concatenate(([self.SOS_IDX], data, [self.EOS_IDX]), axis=0)
        return data
