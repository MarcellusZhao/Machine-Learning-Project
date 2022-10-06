import os
import sys
import numpy as np
import torch
from preprocess.process_raw_data import load_xlsx, frequency, load_eph, assemble_data
import pickle as pkl

from preprocess.generate_dataset import create_dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Step 1:
    Read raw EEG EPH files and save them in numpy array.
"""
subject, group, sex, age, voc, digit_span, acc_deno = load_xlsx('./data/behavioral_info.xlsx')
subjects = load_eph('./rawdata/EPOCHS')
f = open('./rawdata/EPOCHS.pkl', 'wb')
pkl.dump(subjects, f)
f.close()

frequency_sex = frequency(sex)
frequency_voc = frequency(voc)
frequency_digit_span = frequency(digit_span[20:])
print('SEX:')
for k,v in frequency_sex.items():
    print(k, v)
print('VOC:')
for k,v in frequency_voc.items():
    print(k, v)
print('DIGIT:')
for k,v in frequency_digit_span.items():
    print(k, v)


"""
Step 2:
    Create raw dataset (eeg, label) from EEG numpy array.
"""
subject, group, sex, age, voc, digit_span, acc_deno = load_xlsx('./data/behavioral_info.xlsx')

f = open('./rawdata/EPOCHS.pkl', 'rb')
subject_eegs = pkl.load(f)
f.close()

label_map = {}
for i in range(len(subject)):
    label_map[subject[i]] = [sex[i], digit_span[i]] # F - female - 1, H - male - 0

dataset = assemble_data(subject_eegs, label_map) #F (3676.) H (3580.), total 7256
torch.save(dataset, './rawdata/EPOCHS.pth')


"""
Step 3:
    Create splits for cross-validation and final dataset from raw dataset.
    5 folds, 10% as test set.
    You can find splits and dataset at ./data or ./data/training_data.
"""
num_split = 5
ratio_test = 0.1
save_path = './data'
create_dataset('./rawdata/EPOCHS.pth', num_split, ratio_test, save_path)


"""
Step 4:
    Plot EEG signals from the dataset to double check.
"""
dataset = torch.load('./data/EEG_dataset.pth')
data = dataset['dataset']
mean = dataset['means']
stds = dataset['stddevs']
#print(data.shape)
x = [i for i in range(300)]
plt.figure(0)
for i in range(128):
    y = data[0,i,:]
    plt.plot(x, y)

plt.savefig('eeg.png')

plt.figure(1)
data = (data - mean)/stds
for i in range(128):
    y = data[0,i,:]
    plt.plot(x, y)

plt.savefig('eeg-norm.png')