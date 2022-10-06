import os
import sys
import torch
import numpy as np

from lib.guided_backprop import GuidedBackPropagation, run_backprop
from lib.classifiers import classifier_CNN

from lib.mapper import load_xyz, map_function
from lib.options import Options

def main(model_load_path, dataset_path, splits_path, fig_path, save_gif, save_path='./mid_results/grad'):

    dataset = torch.load(dataset_path)
    test_idx = torch.load(splits_path)['splits'][0]['test']
    means = dataset["means"][0].type(torch.FloatTensor)
    stddevs = dataset["stddevs"][0].type(torch.FloatTensor)

    EEGs = dataset['dataset'][test_idx].type(torch.FloatTensor)
    labels = dataset['genders'][test_idx].type(torch.LongTensor)

    net = classifier_CNN(in_channel=128, num_points=300, n_class=2)

    if save_gif:
        try:
            xyz = np.load('data/xyz.npy')
            XY = map_function(xyz)
            print('Load xyz array succeed!')
        except:
            print('Load xyz array fail! Recompute and save!')
            xyz_file = 'data/Biosemi128OK.xyz'
            xyz, channel_name = load_xyz(xyz_file)
            np.save('data/xyz.npy', xyz)
    else:
        XY = None

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    run_backprop(net, model_load_path, EEGs, labels, means, stddevs, save_path, fig_path, plot_gif=save_gif, XY=XY)

if __name__ == "__main__":
    opt, _ = Options().parse(if_print=False)

    save_gif = opt.gif
    dataset_path = opt.eeg_dataset
    splits_path = opt.splits_path
    fig_path = opt.fig_path
    model_load_path = opt.load_path
    main(model_load_path, dataset_path, splits_path, fig_path, save_gif)