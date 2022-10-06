#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18
# Modified: 2021-12-23 by Ren Li

import os, sys
import numpy as np
import cv2
import torch
from torch.nn import ReLU
import torch.nn.functional as F
import imageio
from scipy.interpolate import griddata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot[:, ids] = 1
        return one_hot

    def forward(self, image):
        self.logits = self.model(image)
        pred = torch.argmax(self.logits, dim=-1)
        return pred

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


def run_backprop(net, model_load_path, EEGs, labels, means, stddevs, save_path, fig_path, plot_gif=True, XY=None):
    # Run pre-trained CNN on test trail and obtain gradient by Guided Backprop.
    # Gradient are saved at ./mid_results/grad.
    
    save_path_female = os.path.join(save_path, 'female')
    save_path_male = os.path.join(save_path, 'male')
    if not os.path.exists(save_path_female):
        os.makedirs(save_path_female)
    if not os.path.exists(save_path_male):
        os.makedirs(save_path_male)
    
    # Run pre-trained CNN on test trail and obtain gradient by Guided Backprop.
    for fold in range(5):
        print('Running pretrained CNN on fold %d/5.'%(fold+1))
        model_load_path_fold = os.path.join(model_load_path, 'CNN_split_%d_best.pth'%(fold))
        net.load_state_dict(torch.load(model_load_path_fold))
        net.eval()
        bp = GuidedBackPropagation(model=net)

        for i in range(len(EEGs)):
            eeg = (EEGs[i] - means)/stddevs
            target_class = labels[i]

            eeg = eeg.t().unsqueeze(0)
            pred = bp.forward(eeg)
            correct = (pred==target_class).item()
            bp.backward(ids=target_class)
            guided_grads = bp.generate() # (1, 300, 128)
            guided_grads = guided_grads[0].t().numpy() # (128, 300)
            
            grad_save_path = save_path_female if target_class else save_path_male
            array_name = 'fold_%d_trail_%04d_%s.npy'%(fold, i, str(correct))
            np.save(os.path.join(grad_save_path, array_name), guided_grads)

    # Load saved gradient and average over trails.
    for gender in ['female', 'male']:
        print('Load gradients from ' + gender + '.')
        grad_path = os.path.join(save_path, gender)#'mid_results/grad/' + gender
        grad_files = os.listdir(grad_path)

        list_grad_sign_clip = []
        for f in grad_files:
            correct = f.split('.')[0].split('_')[-1]
            if correct == 'False':
                continue
                
            trail = int(f.split('.')[0].split('_')[3])
            eeg = (EEGs[trail] - means)/stddevs
            eeg = eeg.numpy()

            grad = np.load(os.path.join(grad_path, f))
            grad_sign = grad*np.sign(eeg)
            grad_sign_clip = np.clip(grad_sign, 0, None)

            list_grad_sign_clip.append(grad_sign_clip)

        grad_sign_clip = sum(list_grad_sign_clip)/len(list_grad_sign_clip)
        np.save(os.path.join(save_path, gender + '_grad_sign_clip.npy'), grad_sign_clip)
    
    grad_sign_clip_f = np.load(os.path.join(save_path, 'female_grad_sign_clip.npy'))
    grad_sign_clip_m = np.load(os.path.join(save_path, 'male_grad_sign_clip.npy'))
    grad_sign_clip = np.abs(grad_sign_clip_f + grad_sign_clip_m)

    # Plot spatio-tempolal heat map
    plot_heat_map(grad_sign_clip, fig_path)
    print('Guided Backpropogation is finished! Find the fig at ' + os.path.join(fig_path, 'heat-sign.png'))
    
    # Plot gif
    if plot_gif:
        print('Start generating GIF...')
        generate_gif(EEGs, labels, grad_sign_clip, XY, fig_path, fps=60)
        print('GIF is finished! Find the gif at ' + os.path.join(fig_path, 'topographic.gif'))


def plot_heat_map(grad_sign_clip, fig_path):
    # Plot significance heat map.
    plt.rcParams["figure.figsize"] = 30,15
    x = np.linspace(-50,250, num=300)/512*1000

    fig, ax = plt.subplots()
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(grad_sign_clip, cmap="magma", aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(-90, 480)

    plt.ylabel('Channel', fontsize=45)
    plt.xlabel('Time (ms)', fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, 'heat-sign.png'))

def generate_gif(EEGs, labels, grad_sign_clip, XY, fig_path, fps=60):
    save_image_path = os.path.join(fig_path, 'gif')
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    EEGs_f_sample = EEGs[labels==1].mean(dim=0)
    EEGs_m_sample = EEGs[labels==0].mean(dim=0)

    f_max = EEGs_f_sample.max()
    m_max = EEGs_m_sample.max()
    f_min = EEGs_f_sample.min()
    m_min = EEGs_m_sample.min()

    grad_max = grad_sign_clip.max()
    grad_min = grad_sign_clip.min()

    X_dat = XY[:,0]
    Y_dat = XY[:,1]

    # create x-y points to be used in heatmap
    num_points = 2000
    xi = np.linspace(X_dat.min(), X_dat.max(), num_points)
    yi = np.linspace(Y_dat.min(), Y_dat.max(), num_points)
    xv, yv = np.meshgrid(xi, yi)
    circle = xv**2 + yv**2

    # plot topographic heat maps for gradients and EEG voltage
    for t in range(len(grad_sign_clip[0])):
        print('Plot image for time: %d/300'%(t+1))
        Z_dat = grad_sign_clip[:,t]
        eeg_dat_f = EEGs_f_sample[:,t]
        eeg_dat_m = EEGs_m_sample[:,t]

        plot_topographic(X_dat, Y_dat, Z_dat, xi, yi, circle, 'magma', grad_min, grad_max, save_image_path, 'heat_'+str(t)+'.png')
        plot_topographic(X_dat, Y_dat, eeg_dat_f, xi, yi, circle, 'coolwarm', f_min, f_max, save_image_path, 'eeg_f_'+str(t)+'.png')
        plot_topographic(X_dat, Y_dat, eeg_dat_m, xi, yi, circle, 'coolwarm', m_min, m_max, save_image_path, 'eeg_m_'+str(t)+'.png')

    # Load image sequence for gif generation
    images = []
    for t in range(len(grad_sign_clip[0])):
        heat = cv2.imread(os.path.join(save_image_path, 'heat_'+str(t)+'.png'))
        eeg_m = cv2.imread(os.path.join(save_image_path, 'eeg_m_'+str(t)+'.png'))
        eeg_f = cv2.imread(os.path.join(save_image_path, 'eeg_f_'+str(t)+'.png'))

        img = cv2.hconcat([eeg_f, heat, eeg_m])[:,:,::-1]
        img = imageio.core.util.Array(img)
        images.append(img)

    imageio.mimsave(os.path.join(fig_path, 'topographic.gif'), images, fps=fps)

def plot_topographic(X_dat, Y_dat, Z_dat, xi, yi, circle, color_code, vmin, vmax, save_path, image_name):
    plt.rcParams["figure.figsize"] = 5,5
    zi = griddata((X_dat, Y_dat), Z_dat, (xi[None,:], yi[:,None]), method='cubic')
    zi[circle>1] = float("NAN")

    fig, ax = plt.subplots()
    c = ax.pcolormesh(xi, yi, zi, cmap=color_code, vmin=vmin, vmax=vmax, shading='auto')
    # set the limits of the plot to the limits of the data
    ax.axis([xi.min(), xi.max(), yi.min(), yi.max()])
    cbar = fig.colorbar(c, ax=ax)
    cbar.remove()

    Circle = plt.Circle((0,0),1)
    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(save_path, image_name))
    plt.close('all')