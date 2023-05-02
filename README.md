# Music Humanisation
Maps musical scores to performances, conditioned on instructions. 

We used a NVIDIA T4 GPU to train this model, and recommend similar hardware to replicate our results. We primarily used the `map.yaml` config file.

# Dataset
We recommend pretraining the decoder on a large dataset, then finetuning on a music harmonisation dataset. We used [Giant MIDI](https://github.com/bytedance/GiantMIDI-Piano) to pretrain, then [ASAP](https://github.com/fosfrancesco/asap-dataset/) to finetune. Download the datasets to some location, and modify your config file to point to it.

# Training
Use the config file to change the dataset, architecture, and training procedure. Notably, you can change how often music is generated through training. Once you have prepared some config file named `x.yaml`, run the following to train:
```
python3 src/train_map.py configs/x.yaml
```

Loss, accuracy, and other metrics are logged using `tensorboard`. Some metrics are used from [this research](https://github.com/jason9693/MusicTransformer-pytorch). To view this, run:
```
python3 -m tensorboard.main --logdir log
```
