import os
import pickle
import random

import pandas as pd
import numpy as np

from miditok import REMI
from tqdm import tqdm
from miditoolkit import MidiFile
import utils
from octuple_preprocess import MIDI_to_encoding

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class ASAPDataset(Dataset):
    def __init__(self, cfg, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX):
        self.midi_dir = cfg.get("midi_dir")
        self.tokenizer = tokenizer
        self.octuple = cfg.get("octuple")

        self.dataset_save_path = cfg.get("dataset_save_path")
        self.max_example_len = cfg.get("max_example_len") - 2

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
                    if self.octuple:
                        midi = MidiFile(path)
                        tokens = self.tokenizer(midi)

                        """if len(tokens) + 2 > self.max_example_len:  # + 2 for SOS and EOS tokens
                            continue"""
                    else:
                        midi = MidiFile(path)
                        tokens = self.tokenizer(midi)[0]
                except (IOError, IndexError):
                    continue

                valid_paths.append(path)

            dataset = pd.DataFrame(valid_paths, columns=["path"])
            dataset.to_csv(self.dataset_save_path, index=False)
            return dataset


    def __len__(self):
        return len(self.data)


    def _construct_and_shift(self, tokens, SOS_IDX, EOS_IDX):
        output = []
        output += [[SOS_IDX] * 8]

        for bar in tokens:
            bar_shifted = [i + 3 for i in bar]
            output += [bar_shifted]

        output += [[EOS_IDX]*8]
        return output


    def __getitem__(self, idx):
        path = self.data.iloc[idx][0]

        midi = MidiFile(path)
        tokens = self.tokenizer(midi)

        if self.octuple:
            oct_max_example_len = self.max_example_len // 5
            oct_max_example_len = self.max_example_len
            if oct_max_example_len + 2 <= len(tokens):
                start = random.randrange(0, len(tokens) - oct_max_example_len)
                tokens = tokens[start:start + oct_max_example_len]
                tokens = self._construct_and_shift(tokens, self.SOS_IDX, self.PAD_IDX)
            else:
                tokens = self._construct_and_shift(tokens, self.SOS_IDX, self.PAD_IDX)
                while len(tokens) < oct_max_example_len:
                    tokens.append([self.PAD_IDX]*8)

        else:  # Sample subset of fixed length
            try:
                tokens = tokens[0]
            except IndexError:
                print("Error tokenizing:\n", tokens)
            if self.max_example_len + 2 <= len(tokens):
                start = random.randrange(0, len(tokens) - self.max_example_len)
                tokens = tokens[start:start + self.max_example_len]
                tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]
            else:
                tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]
                while len(tokens) < self.max_example_len:
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


def load_data(cfg):
    SOS_IDX = 0
    EOS_IDX = 1
    PAD_IDX = 2
    if cfg.get("octuple"):
        tokenizer = MIDI_to_encoding
        vocab_size = -1
    else:
        tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX = build_tokenizer()
        vocab_size = len(tokenizer.vocab._token_to_event)
        
    dataset = ASAPDataset(cfg, tokenizer, SOS_IDX, EOS_IDX, PAD_IDX)

    # Create splits
    indices = list(range(len(dataset)))
    if cfg.get("shuffle"):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = cfg.get("dataset_split")
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    batch_size = cfg.get("batch_size")

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), collate_fn=PadCollate(PAD_IDX))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(val_indices), collate_fn=PadCollate(PAD_IDX))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(test_indices), collate_fn=PadCollate(PAD_IDX))

    return train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, vocab_size
