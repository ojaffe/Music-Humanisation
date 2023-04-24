import os
import numpy as np
import torch
import torch.nn.functional as F
#import torchvision
# from custom.config import config
import yaml
from tqdm import tqdm
from octuple_preprocess import encoding_to_MIDI
import random


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def revert_example(example):
    out = []
    for b in example:
        new_b = []
        for n in b:
            if (n - 3) < 0:
                break

            new_b.append(n - 3)

        if len(new_b) == 8:
            out.append(new_b)

    return out


def greedy_decode_octuple(mt, tokenizer, loader, num_heads, SOS_IDX, EOS_IDX, PAD_IDX, global_step, device):
    mt.set_test()

    enc = next(iter(loader))
    enc = enc[0].unsqueeze(0)
    enc = enc.to(device, non_blocking=True, dtype=torch.int)

    prior = [[SOS_IDX]*8]
    dec_in = torch.tensor([prior]).to(device)
    decoded_tokens = prior

    length = 2048
    with torch.no_grad():
        for i in tqdm(range(length)):
            logits = mt(enc, None, dec_in, None, None)
            decoded_example = []
            for out in logits:
                final_out = out[:, out.shape[1]-1, :]
                top_token = torch.argmax(final_out).item()

                decoded_example.append(top_token)

            if top_token - 3 == EOS_IDX:
                break

            dec_in = torch.cat((dec_in, torch.tensor([[decoded_example]]).to(device)), dim=1)

    decoded_tokens_list = dec_in.detach().tolist()[0]
    decoded_tokens_list_rev = revert_example(decoded_tokens_list)
    if len(decoded_tokens_list_rev) == 0:
        return decoded_tokens_list

    try:
        midi = encoding_to_MIDI(decoded_tokens_list_rev)
        midi.dump("bin/gen_{:}.mid".format(global_step))
        print("Successful generation")
    except ValueError:
        print("Generation error")


def greedy_decode(mt, tokenizer, loader, num_heads, SOS_IDX, EOS_IDX, PAD_IDX, global_step, device):
    mt.set_test()

    enc = next(iter(loader))
    enc = enc[0].unsqueeze(0)
    enc = enc.to(device, non_blocking=True, dtype=torch.int)

    enc_pad_mask = build_pad_mask(enc, PAD_IDX)

    prior = [SOS_IDX]
    dec = torch.tensor([prior]).to(device)
    decoded_tokens = prior

    length = 2048
    with torch.no_grad():
        for i in tqdm(range(length)):
            dec_pad_mask = build_pad_mask(dec, PAD_IDX).to(device)
            dec_causal_mask = torch.full((num_heads, dec.shape[1], dec.shape[1]), False).to(device)

            logits = mt(enc, enc_pad_mask, dec, dec_pad_mask, dec_causal_mask)

            logits = logits[:, logits.shape[1]-1, :]
            top_token = torch.argmax(logits).item()

            decoded_tokens.append(top_token)

            if top_token == EOS_IDX:
                break

            dec = torch.cat((dec, torch.tensor([[top_token]]).to(device)), dim=1)

    try:
        midi = tokenizer.tokens_to_midi(enc.detach().tolist(), [(0, False)])
        midi.dump("bin/tgt_{:}.mid".format(global_step))

        midi = tokenizer.tokens_to_midi([decoded_tokens], [(0, False)])
        midi.dump("bin/gen_{:}.mid".format(global_step))
        print("Successful generation")
    except ValueError:
        print("Generation error")


def build_pad_mask(inputs, pad_idx):
    pad_mask = (inputs == pad_idx).bool()

    return pad_mask


def build_causal_mask(inputs, num_heads):
    batch_size, sequence_length = inputs.size()

    causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=inputs.device))
    causal_mask = (causal_mask == 0)
    causal_mask = causal_mask.repeat(batch_size * num_heads, 1, 1).bool()
    return causal_mask


def build_causal_pad_mask(inputs, pad_idx):
    """
    Generates a square causal mask in PyTorch for a batch of inputs, while also masking pad tokens.

    Args:
        inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        pad_idx (int): Index of the pad token in the input tensor.

    Returns:
        torch.Tensor: Causal mask of shape (batch_size, sequence_length, sequence_length).
    """
    # Get the batch size and sequence length
    batch_size, sequence_length = inputs.size()

    # Generate a boolean mask to identify pad tokens
    pad_mask = (inputs == pad_idx)

    # Create a lower triangular matrix with True values below the main diagonal
    causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=inputs.device))

    # Expand the mask to match the batch size
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Mask out the pad tokens
    causal_mask = causal_mask.masked_fill(pad_mask.unsqueeze(1), False)

    return causal_mask


def get_masked_with_pad_tensor(src, trg, pad_token):
    """
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token
    :return:
    """
    size = src.shape[1]
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = torch.equal(src, src_pad_tensor)
    trg_mask = torch.equal(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask


def get_mask_tensor(size):
    """
    :param size: max length of token
    :return:
    """
    # boolean reversing i.e) True * -1 + 1 = False
    seq_mask = ~sequence_mask(torch.arange(1, size + 1), size)
    return seq_mask


def fill_with_placeholder(prev_data: list, max_len: int, fill_val: float):
    placeholder = [fill_val for _ in range(max_len - len(prev_data))]
    return prev_data + placeholder


def pad_with_length(max_length: int, seq: list, pad_val: float):
    """
    :param max_length: max length of token
    :param seq: token list with shape:(length, dim)
    :param pad_val: padding value
    :return:
    """
    pad_length = max(max_length - len(seq), 0)
    pad = [pad_val] * pad_length
    return seq + pad


def append_token(data: torch.Tensor, eos_token):
    start_token = torch.ones((data.size(0), 1), dtype=data.dtype) * eos_token
    end_token = torch.ones((data.size(0), 1), dtype=data.dtype) * eos_token

    return torch.cat([start_token, data, end_token], -1)


def shape_list(x):
    """Shape list"""
    x_shape = x.size()
    x_get_shape = list(x.size())

    res = []
    for i, d in enumerate(x_get_shape):
        if d is not None:
            res.append(d)
        else:
            res.append(x_shape[i])
    return res


def attention_image_summary(name, attn, step=0, writer=None):
    """Compute color image summary.
    Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
    """
    num_heads = attn.size(1)
    # [batch, query_length, memory_length, num_heads]
    image = attn.permute(0, 2, 3, 1)
    image = torch.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = F.pad(image, [0,  -num_heads % 3, 0, 0, 0, 0, 0, 0,])
    image = split_last_dimension(image, 3)
    image = image.max(dim=4).values
    grid_image = torchvision.utils.make_grid(image.permute(0, 3, 1, 2))
    writer.add_image(name, grid_image, global_step=step, dataformats='CHW')


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    x_shape = x.size()
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return torch.reshape(x, x_shape[:-1] + (n, m // n))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


if __name__ == '__main__':

    s = np.array([np.array([1, 2]*50),np.array([1, 2, 3, 4]*25)])

    t = np.array([np.array([2, 3, 4, 5, 6]*20), np.array([1, 2, 3, 4, 5]*20)])
    print(t.shape)

    print(get_masked_with_pad_tensor(100, s, t))

