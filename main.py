"""
In this script, we train our models using full eeg signals,
or truncted eeg signals (window), or single-channel eeg signals.
"""

import os
import sys
import random
import numpy as np
import pickle as pkl
from lib.options import Options
import torch
# Fix random seed for Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from torch.utils.data import DataLoader
from lib.Loader import EEGDataset, Splitter
from lib.classifiers import classifier_GRU, classifier_MLP, classifier_CNN, classifier_LIN, net_trainer

opt, opt_mode = Options().parse()

class_num = 2
split_num = opt.split_num
save_path = opt.save_path
save_model = opt.save_model
channel_idx = opt_mode['channel_idx']
channel_num = opt_mode['channel_num']
eeg_length = opt_mode['eeg_length']

dataset = EEGDataset(opt.eeg_dataset, use_window=(opt.train_mode == 'window'), window_len=eeg_length, window_s=opt.window_s)

# Create loaders for GRU/MLP/CNN/LIN
loaders = {split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=split_num, split_name=split), batch_size=opt.batch_size, drop_last=False, shuffle=True) for split in ["train", "val", "test"]}


if opt.classifier == 'GRU':
	net = classifier_GRU(input_size=channel_num, gru_size=128, output_size=class_num)
elif opt.classifier == 'LIN':
	net = classifier_LIN(input_size=channel_num*eeg_length, n_class=1)
elif opt.classifier == 'MLP':
	net = classifier_MLP(input_size=channel_num*eeg_length, n_class=class_num)
elif opt.classifier == 'CNN':
	net = classifier_CNN(in_channel=channel_num, num_points=eeg_length, n_class=class_num)


accuracy_val, accuracy_test, best_epoch, best_val_test = net_trainer(net, loaders, opt, save_path, classifier_name=opt.classifier, split_num=split_num, channel_idx=channel_idx, save_model=save_model)

print('Best epoch: ', best_epoch, ', Test acc: ', best_val_test)