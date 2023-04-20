import os
import pickle
import random

import pandas as pd
import numpy as np

from miditok import REMI
from tqdm import tqdm
from miditoolkit import MidiFile
import utils

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class ASAPDataset(Dataset):
    def __init__(self, config, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX):
        self.midi_dir = config.midi_dir
        self.tokenizer = tokenizer

        self.dataset_save_path = config.dataset_save_path
        self.max_seq = config.max_seq - 2

        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX

        # Build data
        self.data = self._build_dataset()

    def _build_dataset(self):
        """
        Input, all robotic bars + idxs of most similar bars to tgt
        Target, single tgt bar
        """
        if os.path.exists(self.dataset_save_path):
            return pd.read_csv(self.dataset_save_path)
        else:
            print("Building dataset")
            midi_paths = list(utils.find_files_by_extensions(self.midi_dir, ['.mid', '.midi']))

            valid_paths = []
            for idx, path in tqdm(enumerate(midi_paths)):
                try:
                    midi = MidiFile(path)
                    tokens = self.tokenizer(midi)[0]
                except (IOError, IndexError):
                    continue

                """if len(tokens) + 2 > self.max_seq:  # + 2 for SOS and EOS tokens
                    continue"""

                valid_paths.append(path)

            dataset = pd.DataFrame(valid_paths, columns=["path"])
            dataset.to_csv(self.dataset_save_path, index=False)
            return dataset


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        path = self.data.iloc[idx][0]

        midi = MidiFile(path)
        tokens = self.tokenizer(midi)[0]

        # Sample subset of fixed length
        if self.max_seq + 2 <= len(tokens):
            start = random.randrange(0, len(tokens) - self.max_seq)
            tokens = tokens[start:start + self.max_seq]
            tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]
        else:
            tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]
            while len(tokens) < self.max_seq:
                tokens.append(self.PAD_IDX)

        return torch.tensor(tokens)


class PadCollate:
    def __init__(self, PAD_IDX):
        self.PAD_IDX = PAD_IDX

    def __call__(self, batch):
        tokens = batch
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.PAD_IDX)

        return tokens


def build_tokenizer():
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 128}
    nb_velocities = 32
    additional_tokens = {
        'Chord': False,
        'Rest': False,
        'Tempo': True,
        'Program': False,
        'TimeSignature': True,
        'rest_range': (2, 32),  # (half, 8 beats)
        'nb_tempos': 512,  # nb of tempo bins
        'tempo_range': (1, 400)
    }  # (min, max)

    tokenizer = REMI(pitch_range,
                        beat_res,
                        nb_velocities,
                        additional_tokens,
                        mask=True,
                        pad=True,
                        sos_eos=True,
                        sep=True,)

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    SEP_IDX = 4

    return tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX


def load_data(config):
    tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX = build_tokenizer()
    vocab_size = len(tokenizer.vocab._token_to_event)
    SOS_IDX = 0
    EOS_IDX = 1
    PAD_IDX = 2

    dataset = ASAPDataset(config, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX)

    # Create splits
    indices = list(range(len(dataset)))
    if config.shuffle:
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = config.dataset_split
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    batch_size = config.batch_size

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=PadCollate(PAD_IDX))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(val_indices), collate_fn=PadCollate(PAD_IDX))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(test_indices), collate_fn=PadCollate(PAD_IDX))

    return train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, vocab_size
