import pickle
import os
import sys
from progress.bar import Bar
import utils

from midi_processor.processor import encode_midi, encode_midi_remi
from miditok import REMI


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


def preprocess_midi(path):
    return encode_midi(path)


def preprocess_midi_remi(tokenizer, path):
    return encode_midi_remi(tokenizer, path)


def preprocess_midi_files_under(midi_root, save_dir):
    tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX = build_tokenizer()

    if not os.path.exists(save_dir):
        midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
        os.makedirs(save_dir, exist_ok=True)
        out_fmt = '{}-{}.data'

        for path in Bar('Processing').iter(midi_paths):
            print(' ', end='[{}]'.format(path), flush=True)

            try:
                data = preprocess_midi_remi(tokenizer, path)[0]
            except KeyboardInterrupt:
                print(' Abort')
                return
            except EOFError:
                print('EOF Error')
                return
            except IOError:
                print('IO Error')
                continue

            with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
                pickle.dump(data, f)

    return tokenizer


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
