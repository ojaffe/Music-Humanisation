from model import MusicTransformer
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data
import custom
from preprocess import preprocess_midi_files_under
from midi_processor.processor import decode_midi
from dataset import load_data

import os
import utils
import datetime
import time
import shutil
import argparse
from utils import load_config

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(cfg_file):
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    train_loader, val_loader, test_loader, tokenizer, PAD_IDX, SOS_IDX, EOS_IDX, vocab_size = load_data(cfg)

    mt = MusicTransformer(
                embedding_dim=cfg.get("embedding_dim"),
                vocab_size=vocab_size,
                num_layer=cfg.get("num_layers"),
                max_seq=cfg.get("max_seq"),
                dropout=cfg.get("dropout"),
                debug=cfg.get("debug"), 
                loader_path=cfg.get("load_path"),
                h=cfg.get("h"),
                PAD_IDX=PAD_IDX
    ).to(device)
    #mt.load_state_dict(torch.load("saved_models/train-60.pth"))

    learning_rate = cfg.get("l_r")
    opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.get("weight_decay"))
    scheduler = CustomSchedule(cfg.get("embedding_dim"), optimizer=opt)

    # init metric set
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

    global_step = 0
    for e in range(cfg.get("epochs")):
        print(e)

        for batch_idx, batch in tqdm(enumerate(train_loader)):
            scheduler.optimizer.zero_grad()

            tokens = batch
            tokens = tokens.to(device, non_blocking=True, dtype=torch.int)
            src = tokens[:, :-1]
            tgt = tokens[:, 1:]

            start_time = time.time()
            mt.set_train()
            sample = mt.forward(src)
            metrics = metric_set(sample, tgt)
            loss = metrics['loss']
            loss.backward()
            scheduler.step()
            end_time = time.time()

            nlll = nllloss(sample.reshape(-1, sample.shape[-1]), tgt.reshape(-1).long()).item()

            if cfg.get("debug"):
                print("[Loss]: {}".format(loss))
                print(src.shape[1], torch.cuda.max_memory_allocated() / 1e9)

            train_summary_writer.add_scalar('loss', metrics['loss'], global_step=global_step)
            train_summary_writer.add_scalar('nlll_loss', nlll, global_step=global_step)
            train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=global_step)
            train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=global_step)
            train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=global_step)

            # Generate example
            if batch_idx % 50 == 0 and batch_idx != 0:
                mt.set_test()

                inputs = np.array([[SOS_IDX]])
                inputs = torch.from_numpy(inputs).to(device)
                with torch.no_grad():
                    result = mt(inputs, length=2048)

                if EOS_IDX in result:
                    eos_idx = result.index(EOS_IDX)
                    result = result[:eos_idx]
                    result.append(EOS_IDX)

                try:
                    midi = tokenizer.tokens_to_midi([result], [(0, False)])
                    midi.dump("bin/{:}.mid".format(global_step))
                    print("Successful generation")
                except ValueError:
                    print("Generation error")

            # Validation loop
            if batch_idx % 50 == 0:
                mt.set_eval()
                val_loss = 0
                val_nlll = 0
                val_acc = 0
                for batch in tqdm(val_loader):
                    tokens = batch
                    tokens = tokens.to(device, non_blocking=True, dtype=torch.int)
                    src = tokens[:, :-1]
                    tgt = tokens[:, 1:]

                    with torch.no_grad():
                        eval_preiction, weights = mt.forward(src)

                    eval_metrics = metric_set(eval_preiction, tgt)
                    nlll = nllloss(eval_preiction.reshape(-1, eval_preiction.shape[-1]), tgt.reshape(-1).long()).item()
                    
                    if batch_idx == 0:
                        train_summary_writer.add_histogram("source_analysis", src, global_step=e)
                        train_summary_writer.add_histogram("target_analysis", tgt, global_step=e)
                        #for i, weight in enumerate(weights):
                            #attn_log_name = "attn/layer-{}".format(i)
                            #utils.attention_image_summary(
                                #attn_log_name, weight, step=global_step, writer=eval_summary_writer)

                    val_loss += eval_metrics['loss']
                    val_acc += eval_metrics['accuracy']
                    val_nlll += nlll

                val_loss_avg = val_loss / len(val_loader)
                val_acc_avg = val_acc / len(val_loader)
                val_nlll_avg = val_nlll / len(val_loader)
                        
                eval_summary_writer.add_scalar('loss', val_loss_avg, global_step=global_step)
                eval_summary_writer.add_scalar('loss_nlll', val_nlll_avg, global_step=global_step)
                eval_summary_writer.add_scalar('accuracy', val_acc_avg, global_step=global_step)
                eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=global_step)

                print('\n====================================================')
                print('Epoch/Batch: {}/{}'.format(e, batch_idx))
                print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
                print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))

            torch.cuda.empty_cache()
            global_step += 1

        torch.save(mt.state_dict(), cfg.get("model_dir")+'/train-{}.pth'.format(e))

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