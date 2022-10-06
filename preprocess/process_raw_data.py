import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import sys
import torch

def load_xlsx(path):
    """Load xlsx and convert it to the metrics system."""
    xl_file = pd.ExcelFile(path)
    dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names} # Feuil1/Feuil2/Feuil3
    variables = dfs['Feuil1']
    subject, group, sex, age, voc, digit_span, acc_deno, rt_deno = variables.loc[:,'SUBJECT'], variables.loc[:,'GROUP'], variables.loc[:,'SEX'], variables.loc[:,'AGE'], variables.loc[:,'VOC NB'], variables.loc[:,'MDC OD'], variables.loc[:,'Acc_DENO'], variables.loc[:,'RT_DENO']

    subject = subject.values.tolist()
    group = group.values.tolist()
    sex = sex.values.tolist()
    age = age.values.tolist()
    voc = voc.values.tolist()
    digit_span = digit_span.values.tolist()
    acc_deno = acc_deno.values.tolist()
    rt_deno = rt_deno.values.tolist()
    return subject, group, sex, age, voc, digit_span, acc_deno, rt_deno

def frequency(data):
    dic = {}
    for d in data:
        if d in dic:
            dic[d] += 1
        else:
            dic[d] = 1
    
    dic = OrderedDict(sorted(dic.items()))
    return dic

def load_eph(path):
    files_eph = sorted(os.listdir(path))
    print('#files:', len(files_eph))
    subjects = {}
    for file_eph in files_eph:
        print('processing ', file_eph)
        subject = file_eph.split('_')[0]
        if subject not in subjects:
            subjects[subject] = np.zeros((0, 128, 300))

        skip_flag = False
        f = open(os.path.join(path, file_eph), 'r')
        lines = f.readlines()
        lines = lines[1:]
        if len(lines) != 300:
            continue

        signals = np.zeros((1, 128, 300))
        #print('#lines:', len(lines)) #300
        for i in range(len(lines)):
            line = lines[i].split()
            signal = list(map(float, line))

            if len(signal) != 128:
                skip_flag = True
                break

            signals[0,:,i] = signal
            #print('#column:', len(signal)) # 128

        f.close()

        if skip_flag:
            continue

        subjects[subject] = np.concatenate([subjects[subject], signals], axis=0)

    for k,v in subjects.items():
        print(k, len(v))

    return subjects

def assemble_data(subject_eegs, label_map):
    eegs = torch.zeros(0,128,300)
    genders = torch.zeros(0)
    digit_spans = torch.zeros(0)
    subject_name = []

    for subject, eeg in subject_eegs.items():
        print('assemble subject ' + subject)
        sex, ds = label_map[subject]
        gender = [1 if sex == 'F' else 0]*len(eeg)
        digit_span = [ds]*len(eeg)
        subject_name += [subject]*len(eeg)

        eegs = torch.cat([eegs, torch.FloatTensor(eeg)], dim=0)
        genders = torch.cat([genders, torch.LongTensor(gender)], dim=0)
        digit_spans = torch.cat([digit_spans, torch.FloatTensor(digit_span)], dim=0)

    print(len(eegs), len(genders), len(digit_spans), len(subject_name))
    print((genders==1).type(torch.FloatTensor).sum(), (genders==0).type(torch.FloatTensor).sum())
    dataset = {'dataset': eegs,
               'genders': genders,
               'digit_spans': digit_spans,
               'subject': subject_name}

    return dataset
