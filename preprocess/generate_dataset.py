import os
import sys
import torch
import numpy as np
import random

def cross_validation(num_samples, ratio=0.1, seed=42):
    num_val_set = int(1/ratio)
    num_val_sample = int(num_samples*ratio)
    idx = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_sets = []
    val_sets = []
    sample_set = set(idx.tolist())
    for i in range(num_val_set):
        val_set = idx[i*num_val_sample:(i+1)*num_val_sample].tolist()
        train_set = list(sample_set-set(val_set))
        val_sets.append(val_set)
        train_sets.append(train_set)
    print('%d train/val sets are created.'%num_val_set)
    return train_sets, val_sets

def create_split(data, save_path, num_split, ratio_test, seed=42, debug=False):
    splits = {'splits':{}}
    for i in range(num_split):
        splits['splits'][i] = {'train':[], 'val':[], 'test':[]}

    len_data = len(data)
    num_test = int(len_data*ratio_test)
    _range = [i for i in range(len_data)]
    random.seed(seed)
    random.shuffle(_range)

    test_idx = _range[:num_test]
    rest_idx = _range[num_test:]

    train_sets, val_sets = cross_validation(len(rest_idx), ratio=1.0/num_split)
    for i in range(num_split):
        splits['splits'][i]['test'] = test_idx

        train_set, val_set = train_sets[i], val_sets[i]
        train_idx = [rest_idx[idx] for idx in train_set]
        val_idx = [rest_idx[idx] for idx in val_set]
        splits['splits'][i]['train'] = train_idx
        splits['splits'][i]['val'] = val_idx
        print('train_idx ', len(train_idx), ', val_idx ', len(val_idx))

        if debug:
            assert len(train_idx) + len(val_idx) == len(rest_idx)
            assert len(train_idx) == len(set(train_idx))
            assert len(val_idx) == len(set(val_idx))
            assert len(rest_idx) == len(set(rest_idx))
            assert set(train_idx).union(set(val_idx)) == set(rest_idx)
            assert len(set(train_idx) - set(val_idx)) == len(train_idx)

    torch.save(splits, save_path)
    return splits

def get_mean_std(data, idx):
    data_train = data[idx]
    #print('data_train: ', data_train.shape)
    mean = torch.mean(data, dim=(0,2), keepdim=True)
    std = torch.std(data, unbiased=False, dim=(0,2), keepdim=True)
    return mean, std

def create_dataset(raw_data_path, num_split, ratio_test, save_path):

    dataset = torch.load(raw_data_path)

    split_save_path = os.path.join(save_path, 'splits.pth')
    splits = create_split(dataset['dataset'], split_save_path, num_split, ratio_test, debug=True)

    means, stds = get_mean_std(dataset['dataset'], splits['splits'][0]['train']+splits['splits'][0]['val'])
    print(means.shape, stds.shape)
    dataset['means'] = means
    dataset["stddevs"] = stds

    dataset_save_path = os.path.join(save_path, 'EEG_dataset.pth')
    torch.save(dataset, dataset_save_path)
    return

def get_data_idx(label_subject, target_subject):
    target = set(target_subject)
    idx = []
    for i in range(len(label_subject)):
        if label_subject[i] in target:
            idx.append(i)
    
    return idx

def create_split_by_subject(dataset_path, num_split, ratio_test, save_path, seed=11, debug=True):
    splits = {'splits':{}}
    for i in range(num_split):
        splits['splits'][i] = {'train':[], 'val':[], 'test':[]}

    dataset = torch.load(dataset_path)
    subject = dataset['subject']
    subject = list(set(subject))
    #print(len(subject))

    len_data = len(subject)
    num_test = int(len_data*ratio_test)
    _range = [i for i in range(len_data)]
    random.seed(seed)
    random.shuffle(_range)
    test_idx = _range[:num_test]
    rest_idx = _range[num_test:]

    test_subject = [subject[idx] for idx in test_idx]
    rest_subject = [subject[idx] for idx in rest_idx]
    print(test_subject)

    test_idx = get_data_idx(dataset['subject'], test_subject)

    train_sets, val_sets = cross_validation(len(rest_subject), ratio=1.0/num_split)
    for i in range(num_split):
        splits['splits'][i]['test'] = test_idx

        train_set, val_set = train_sets[i], val_sets[i]
        train_subject = [rest_subject[idx] for idx in train_set]
        val_subject = [rest_subject[idx] for idx in val_set]
        print(val_subject)
        train_idx = get_data_idx(dataset['subject'], train_subject)
        val_idx = get_data_idx(dataset['subject'], val_subject)
        splits['splits'][i]['train'] = train_idx
        splits['splits'][i]['val'] = val_idx

        print('train_idx ', len(train_idx), ', val_idx ', len(val_idx), ', test_idx ', len(test_idx))

        if debug:
            assert len(train_idx) + len(val_idx) + len(test_idx) == len(dataset['subject'])
            assert len(train_idx) == len(set(train_idx))
            assert len(val_idx) == len(set(val_idx))
            assert len(set(train_idx) - set(val_idx)) == len(train_idx)


    split_save_path = os.path.join(save_path, 'splits_by_subject.pth')
    torch.save(splits, split_save_path)

    means, stds = get_mean_std(dataset['dataset'], splits['splits'][0]['train']+splits['splits'][0]['val']+splits['splits'][0]['test'])
    print(means.shape, stds.shape)
    dataset['means'] = means
    dataset["stddevs"] = stds

    dataset_save_path = os.path.join(save_path, 'EEG_dataset_by_subject.pth')
    torch.save(dataset, dataset_save_path)
    return