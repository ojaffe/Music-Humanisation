from model import MusicTransformer
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data
import custom
from dataset import load_data

from transformer.model.transformer import Transformer
from utils import load_config, build_causal_mask, build_pad_mask, greedy_decode, greedy_decode_octuple, set_seed

import os
import utils
import datetime
import time
import shutil
import argparse
import math

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    set_seed(seed=cfg.get("seed"))

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, vocab_size = load_data(cfg)

    mt = Transformer(cfg,
                     octuple=cfg.get("octuple"),
                     max_example_len=cfg.get("max_example_len"),
                     vocab_size=vocab_size,
                     SOS_IDX=SOS_IDX,
                     EOS_IDX=EOS_IDX,
                     PAD_IDX=PAD_IDX,
                     device=device).to(device)
    if False:
        mt.load_state_dict(torch.load("saved_models/train-3-0.pth"))
    else:
        for p in mt.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    learning_rate = cfg.get("l_r")
    opt = optim.Adam(mt.parameters(), lr=cfg.get("l_r"), betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.get("weight_decay"))
    #scheduler = CustomSchedule(cfg.get("d_model"), optimizer=opt)

    # init metric set
    if cfg.get("octuple"):
        metric_set = [MetricsSet({
            'accuracy': CategoricalAccuracy(),
            'loss': SmoothCrossEntropyLoss(cfg.get("label_smooth"), x, PAD_IDX),
            'bucket':  LogitsBucketting(vocab_size)
        }) for x in cfg.get("octuple_em_sizes")]
    else:
        metric_set = MetricsSet({
        'accuracy': CategoricalAccuracy(),
        'loss': SmoothCrossEntropyLoss(cfg.get("label_smooth"), vocab_size, PAD_IDX),
        'bucket':  LogitsBucketting(vocab_size)
    })

    # tensorboard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = cfg.get("log_dir")+cfg.get("experiment")+'/'+current_time+'/train'
    eval_log_dir = cfg.get("log_dir")+cfg.get("experiment")+'/'+current_time+'/eval'
    nllloss = torch.nn.NLLLoss(ignore_index=PAD_IDX)

    if cfg.get("clear_log") and os.path.exists(cfg.get("log_dir")):
        shutil.rmtree(cfg.get("log_dir"))
    train_summary_writer = SummaryWriter(train_log_dir)
    eval_summary_writer = SummaryWriter(eval_log_dir)

    if not os.path.exists(cfg.get("model_dir")):
        os.makedirs(cfg.get("model_dir"))

    global_step = 0
    for e in range(cfg.get("epochs")):
        print(e)

        train_loss = 0
        avg_loss = []
        avg_acc = []
        avg_time = []
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            #scheduler.optimizer.zero_grad()
            opt.zero_grad()

            tokens = batch
            tokens = tokens.to(device, non_blocking=True, dtype=torch.int)
            dec_in = tokens[:, :-1]
            dec_out = tokens[:, 1:]

            if cfg.get("octuple"):
                enc_pad_mask = build_pad_mask(tokens[:, :, 0], PAD_IDX)
                dec_pad_mask = build_pad_mask(dec_in[:, :, 0], PAD_IDX)
                dec_causal_mask = build_causal_mask(dec_in[:, :, 0], cfg.get("num_heads"))

                start_time = time.time()
                mt.set_train()
                sample = mt.forward(tokens, enc_pad_mask, dec_in, dec_pad_mask, dec_causal_mask)

                loss_oct = 0
                acc_oct = []
                for idx, out in enumerate(sample):
                    dec_out_token = dec_out[:, :, idx]
                    metrics = metric_set[idx](out, dec_out_token)

                    train_loss += metrics['loss']

                    loss_oct += metrics['loss']
                    acc_oct.append(metrics['accuracy'])

                pass_time = time.time() - start_time

                acc_oct_avg = sum(acc_oct) / len(acc_oct)
                avg_loss.append(loss_oct)
                avg_acc.append(acc_oct_avg)
                avg_time.append(pass_time)
            else:
                enc_pad_mask = build_pad_mask(tokens, PAD_IDX)
                dec_pad_mask = build_pad_mask(dec_in, PAD_IDX)
                dec_causal_mask = build_causal_mask(dec_in, cfg.get("num_heads"))

                start_time = time.time()
                mt.set_train()
                sample = mt.forward(tokens, enc_pad_mask, dec_in, dec_pad_mask, dec_causal_mask)
                metrics = metric_set(sample, dec_out)
                train_loss += metrics['loss']
                pass_time = time.time() - start_time

                avg_loss.append(metrics['loss'])
                avg_acc.append(metrics['accuracy'])
                avg_time.append(pass_time)

            # Gradient accumalation
            if batch_idx % cfg.get("grad_accum") == 0 and batch_idx != 0:
                if train_loss.item() == None or math.isnan(train_loss.item()):
                    print("Loss error")
                    train_loss = 0
                    avg_loss = []
                    avg_acc = []
                    avg_time = []
                else:
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(mt.parameters(), 0.1)
                    opt.step()
                    #scheduler.step()

                    train_summary_writer.add_scalar('loss', sum(avg_loss) / len(avg_loss), global_step=global_step)
                    train_summary_writer.add_scalar('accuracy', sum(avg_acc) / len(avg_acc), global_step=global_step)
                    #train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=global_step)
                    train_summary_writer.add_scalar('iter_p_sec', sum(avg_time) / len(avg_time), global_step=global_step)

                    train_loss = 0
                    avg_loss = []
                    avg_acc = []
                    avg_time = []                    

            if cfg.get("debug"):
                print("[Loss]: {}".format(train_loss))
                print(tokens.shape[1], torch.cuda.max_memory_allocated() / 1e9)

            # Generate example
            if global_step % 200 == 0 and batch_idx != 0:
                if cfg.get("octuple"):
                    greedy_decode_octuple(mt, tokenizer, test_loader, cfg.get("num_heads"), SOS_IDX, EOS_IDX, PAD_IDX, global_step, device)
                else:
                    greedy_decode(mt, tokenizer, test_loader, cfg.get("num_heads"), SOS_IDX, EOS_IDX, PAD_IDX, global_step, device)

            # Validation loop
            if global_step % 200 == 0 and batch_idx != 0:
                mt.set_eval()
                val_loss = 0
                val_acc = 0
                for batch in tqdm(val_loader):
                    tokens = batch
                    tokens = tokens.to(device, non_blocking=True, dtype=torch.int)
                    dec_in = tokens[:, :-1]
                    dec_out = tokens[:, 1:]

                    if cfg.get("octuple"):
                        enc_pad_mask = build_pad_mask(tokens[:, :, 0], PAD_IDX)
                        dec_pad_mask = build_pad_mask(dec_in[:, :, 0], PAD_IDX)
                        dec_causal_mask = build_causal_mask(dec_in[:, :, 0], cfg.get("num_heads"))

                        start_time = time.time()
                        mt.set_eval()
                        with torch.no_grad():
                            sample = mt.forward(tokens, enc_pad_mask, dec_in, dec_pad_mask, dec_causal_mask)

                        loss = 0
                        acc = []
                        for idx, out in enumerate(sample):
                            dec_out_token = dec_out[:, :, idx]
                            metrics = metric_set[idx](out, dec_out_token)
                            loss += metrics['loss']
                            acc.append(metrics['accuracy'])

                        end_time = time.time()

                        acc = sum(acc) / len(acc)
                    else:
                        enc_pad_mask = build_pad_mask(tokens, PAD_IDX)
                        dec_pad_mask = build_pad_mask(dec_in, PAD_IDX)
                        dec_causal_mask = build_causal_mask(dec_in, cfg.get("num_heads"))

                        start_time = time.time()
                        mt.set_eval()
                        with torch.no_grad():
                            sample = mt.forward(tokens, enc_pad_mask, dec_in, dec_pad_mask, dec_causal_mask)
                        metrics = metric_set(sample, dec_out)
                        loss = metrics['loss']
                        end_time = time.time()

                        acc = metrics['accuracy']
                    
                    #if batch_idx == 0:
                        #train_summary_writer.add_histogram("source_analysis", dec_in, global_step=e)
                        #train_summary_writer.add_histogram("target_analysis", dec_out, global_step=e)
                        #for i, weight in enumerate(weights):
                            #attn_log_name = "attn/layer-{}".format(i)
                            #utils.attention_image_summary(
                                #attn_log_name, weight, step=global_step, writer=eval_summary_writer)

                    val_loss += loss.item()
                    val_acc += acc

                val_loss_avg = val_loss / len(val_loader)
                val_acc_avg = val_acc / len(val_loader)
                        
                eval_summary_writer.add_scalar('loss', val_loss_avg, global_step=global_step)
                eval_summary_writer.add_scalar('accuracy', val_acc_avg, global_step=global_step)
                #eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=global_step)

            torch.cuda.empty_cache()
            global_step += 1

            if batch_idx % 1000 == 0:
                torch.save(mt.state_dict(), cfg.get("model_dir")+'/train-{}-{}.pth'.format(e, batch_idx))

    torch.save(mt.state_dict(), cfg.get("model_dir")+'/final.pth'.format(global_step))
    eval_summary_writer.close()
    train_summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)