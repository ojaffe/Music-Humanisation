from model import MusicTransformer
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data
import custom
from midi_processor.processor import decode_midi

import os
import utils
import datetime
import time
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Data(config.pickle_dir, config.token_sos, config.token_eos)

mt = MusicTransformer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq=config.max_seq,
            dropout=config.dropout,
            h=config.h,
            flash=False,
            debug=config.debug, 
            loader_path=config.load_path,
            PAD_IDX=config.pad_token
).to(device)

learning_rate = config.l_r
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=config.weight_decay)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})

# tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = config.log_dir+config.experiment+'/'+current_time+'/train'
eval_log_dir = config.log_dir+config.experiment+'/'+current_time+'/eval'

if config.clear_log and os.path.exists(config.log_dir):
    shutil.rmtree(config.log_dir)
train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
print(">> Train start...")
global_step = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in tqdm(range(len(dataset.files) // config.batch_size)):
        scheduler.optimizer.zero_grad()

        src, tgt = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
        src = torch.from_numpy(src).contiguous().to(device, non_blocking=True, dtype=torch.int)
        tgt = torch.from_numpy(tgt).contiguous().to(device, non_blocking=True, dtype=torch.int)

        start_time = time.time()
        mt.train()
        sample = mt.forward(src)
        metrics = metric_set(sample, tgt)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        if config.debug:
            print("[Loss]: {}".format(loss))
            print(src.shape, torch.cuda.max_memory_allocated() / 1e9)

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=global_step)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=global_step)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=global_step)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=global_step)

        # result_metrics = metric_set(sample, tgt)
        if b % 50 == 0:
            mt.set_eval()
            src, tgt = dataset.slide_seq2seq_batch(2, config.max_seq, 'eval')
            src = torch.from_numpy(src).contiguous().to(device, dtype=torch.int)
            tgt = torch.from_numpy(tgt).contiguous().to(device, dtype=torch.int)

            with torch.no_grad():
                eval_preiction, weights = mt.forward(src)

            eval_metrics = metric_set(eval_preiction, tgt)
            torch.save(mt.state_dict(), args.model_dir+'/train-{}.pth'.format(e))
            if b == 0:
                train_summary_writer.add_histogram("source_analysis", src, global_step=e)
                train_summary_writer.add_histogram("target_analysis", tgt, global_step=e)
                """for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(
                        attn_log_name, weight, step=global_step, writer=eval_summary_writer)"""

            # Generate example
            if b % 50 == 0 and b != 0:
                mt.set_test()
                config.threshold_len = 500

                inputs = np.array([[config.event_dim + 1]])
                inputs = torch.from_numpy(inputs).to(device)
                with torch.no_grad():
                    result = mt(inputs, length=2500)

                if config.token_eos in result:
                    eos_idx = result.index(config.token_eos)
                    result = result[:eos_idx]
                    result.append(config.token_eos)
            
                try:
                    decode_midi(result, file_path="bin/{:}.mid".format(global_step))
                    print("Successful generation")
                except ValueError:
                    print("Generation error")
                    pass

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=global_step)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=global_step)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=global_step)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
            
        torch.cuda.empty_cache()
        global_step += 1

torch.save(mt.state_dict(), args.model_dir+'/final.pth'.format(global_step))
eval_summary_writer.close()
train_summary_writer.close()
