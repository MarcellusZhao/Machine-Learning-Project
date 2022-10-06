import os
import random
import numpy as np
import pickle as pkl
import torch
from scipy import stats
from preprocess.process_raw_data import load_xlsx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##############################################################
# tANOVA analysis
# For more details, see
# Topographic ERP Analyses: A Step-by-Step Tutorial Review
# by Micah M. Murray, Denis Brunet, Christoph M. Michel
##############################################################
def tANOVA(dataset, test_idx, fig_path, save_mid_result=False):
    print('Run tANOVA.')
    random.seed(0)
    means = dataset["means"][0].type(torch.FloatTensor)
    stddevs = dataset["stddevs"][0].type(torch.FloatTensor)

    EEGs = dataset['dataset'][test_idx].type(torch.FloatTensor)
    labels = dataset['genders'][test_idx].type(torch.LongTensor)

    EEGs = (EEGs-means)/stddevs # (N, 128, 300)

    ERP_f = EEGs[labels==1].mean(dim=0)
    ERP_m = EEGs[labels==0].mean(dim=0)

    ERP_f = ERP_f - ERP_f.mean(dim=0, keepdim=True)
    ERP_m = ERP_m - ERP_m.mean(dim=0, keepdim=True)
    std_f = torch.std(ERP_f, dim=0, keepdim=True)
    std_m = torch.std(ERP_m, dim=0, keepdim=True)
    ERP_f /= std_f
    ERP_m /= std_m

    # Global dissimilarity (DISS)
    DISS = torch.sqrt(((ERP_f-ERP_m)**2).mean(0))

    time = len(DISS)
    probability = []
    for t in range(time):
        print('Calculating tANOVA for time: %d/300'%(t+1))
        diss = DISS[t]
        diss_shuffle = []
        for n in range(1000):
            idx = [i for i in range(len(EEGs))]
            random.shuffle(idx)
            labels_shuffled = labels[idx]
            ERP_f = EEGs[labels_shuffled==1, :, t].mean(dim=0)
            ERP_m = EEGs[labels_shuffled==0, :, t].mean(dim=0)

            ERP_f = ERP_f - ERP_f.mean(dim=0, keepdim=True)
            ERP_m = ERP_m - ERP_m.mean(dim=0, keepdim=True)
            std_f = torch.std(ERP_f, dim=0, keepdim=True)
            std_m = torch.std(ERP_m, dim=0, keepdim=True)
            ERP_f /= std_f
            ERP_m /= std_m
            diss_t = torch.sqrt(((ERP_f-ERP_m)**2).mean(0))

            diss_shuffle.append(diss_t)

        diss_shuffle = sorted(diss_shuffle, reverse=True)
        for n in range(1000):
            if diss_shuffle[n] < diss:
                probability.append(1 - n*1.0/1000)
                break

            if n == 999:
                probability.append(0.0)

    if save_mid_result:
        if not os.path.exists('./mid_results'):
            os.makedirs('./mid_results')
        f = open('mid_results/t-ANOVA-DISS.pkl', 'wb')
        pkl.dump({'prob':probability}, f)
        f.close()
    
    prob = [ i > 0.99 for i in probability]
    prob = np.array(prob).reshape(1,-1)

    plt.rcParams["figure.figsize"] = 30,10
    cmap = matplotlib.colors.ListedColormap(['white', 'cyan'])
    x = np.linspace(-50,250, num=300)/512*1000
    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    fig, (ax,ax1) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [0.2, 1]})
    ax.imshow(prob, cmap=cmap, aspect=20, extent=extent)
    ax.set_yticks([])

    ax1.plot(x, probability, linewidth=8)

    size = 40
    plt.xlabel('Time (ms)', fontsize=size)
    plt.ylabel('tANOVA (1-p)', fontsize=size)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.xlim([-90, 480])
    plt.tight_layout()
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, 'tANOVA.png'))


##############################################################
# T-test analysis 
# Calculate the p-value for each spatio-temporal point.
# Points with p < 0.001 are marked in red.
# If p < 0.001, reject the null hypothesis of equal group means.
##############################################################
def t_test(dataset, fig_path, save_mid_result=False):
    print('Run T-test.')
    random.seed(0)

    EEGs = dataset['dataset'].type(torch.FloatTensor)
    labels = dataset['genders'].type(torch.LongTensor)

    EEG_f = EEGs[labels==1]
    EEG_m = EEGs[labels==0]
    idx = [i for i in range(len(EEG_f))]
    random.shuffle(idx)
    idx = idx[:len(EEG_m)]
    EEG_f = EEG_f[idx]

    p_values = np.zeros((128,300)) # probability from same populations
    for c in range(128):
        print('Calculating t-test for channel: %d/128'%(c+1))
        for t in range(300):
            rvs1 = EEG_f[:, c, t]
            rvs2 = EEG_m[:, c, t]
            t_value, p_value = stats.ttest_ind(rvs1, rvs2, equal_var=False)
            p_values[c, t] = p_value

    p_values = 1 - p_values
    if save_mid_result:
        if not os.path.exists('./mid_results'):
            os.makedirs('./mid_results')
        np.save('mid_results/t-test.npy', p_values)

    #p_values = np.load('mid_results/t-test.npy')
    p_values[p_values>=0.999] = 1
    p_values[p_values<0.999] = 0
    plt.rcParams["figure.figsize"] = 30,10
    x = np.linspace(-50,250, num=300)/512*1000
    fig, ax = plt.subplots()

    cmap = matplotlib.colors.ListedColormap(['white', 'red'])

    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
    ax.imshow(p_values, cmap=cmap, aspect=120, extent=extent)
    ax.set_yticks([])
    ax.set_xlim(-90, 480)

    plt.xlabel('Time (ms)', fontsize=45)
    plt.ylabel('t-test (1-p)', fontsize=45)
    plt.xticks(fontsize=45)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, 't-test.png'))


##############################################################
# H-test (Kruskal-Wallis test) analysis 
# Calculate means and stds of reaction time and accuracy.
# Also calculate p-values for accuracy and reaction time.
# If p >> 0.05, keep the null hypothesis of equal group medians.
##############################################################
def h_test(behavioral_info):
    print('Run H-test.')
    subject, group, sex, age, voc, digit_span, acc_deno, rt_deno = load_xlsx(behavioral_info)

    acc_deno = np.array(acc_deno)
    rt_deno = np.array(rt_deno)
    
    idx_f = [i for i in range(len(sex)) if sex[i] == 'F']
    idx_m = [i for i in range(len(sex)) if sex[i] == 'H']

    acc_deno_m = acc_deno[idx_m]
    acc_deno_f = acc_deno[idx_f]
    rt_deno_m = rt_deno[idx_m]
    rt_deno_f = rt_deno[idx_f]

    acc_deno_m = acc_deno_m[np.argwhere(~np.isnan(acc_deno_m))].reshape(-1)
    acc_deno_f = acc_deno_f[np.argwhere(~np.isnan(acc_deno_f))].reshape(-1)
    rt_deno_m = rt_deno_m[np.argwhere(~np.isnan(rt_deno_m))].reshape(-1)
    rt_deno_f = rt_deno_f[np.argwhere(~np.isnan(rt_deno_f))].reshape(-1)

    h_value_acc, p_value_acc = stats.kruskal(acc_deno_m, acc_deno_f)
    h_value_rt, p_value_rt = stats.kruskal(rt_deno_m, rt_deno_f)

    acc_m_mean = np.mean(acc_deno_m)
    acc_f_mean = np.mean(acc_deno_f)

    acc_m_std = np.std(acc_deno_m)
    acc_f_std = np.std(acc_deno_f)

    rt_m_mean = np.mean(rt_deno_m)
    rt_f_mean = np.mean(rt_deno_f)

    rt_m_std = np.std(rt_deno_m)
    rt_f_std = np.std(rt_deno_f)

    print('Accuracy: F - %0.3f(%0.3f), M - %0.3f(%0.3f)'%(acc_f_mean, acc_f_std, acc_m_mean, acc_m_std))
    print('Reaction Time: F - %0.3f(%0.3f), M - %0.3f(%0.3f)'%(rt_f_mean, rt_f_std, rt_m_mean, rt_m_std))
    print('p-values (H-test): Accuracy - %0.3f, Reation Time - %0.3f'%(p_value_acc, p_value_rt))