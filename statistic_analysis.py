import torch
import os
from lib.options import Options
from lib.analysis import tANOVA, t_test, h_test


def main(analysis_type, dataset_path, splits_path, fig_path, save_mid_result=False):
    dataset = torch.load(dataset_path)
    test_idx = torch.load(splits_path)['splits'][0]['test']
    if analysis_type == 'tANOVA':
        tANOVA(dataset, test_idx, fig_path, save_mid_result=save_mid_result)
        print('tANOVA is finished. Find the fig at ' + os.path.join(fig_path, 'tANOVA.png'))
    elif analysis_type == 'ttest':
        t_test(dataset, fig_path, save_mid_result=save_mid_result)
        print('T-test is finished. Find the fig at ' + os.path.join(fig_path, 't-test.png'))
    elif analysis_type == 'htest':
        behavioral_info = './data/behavioral_info.xlsx'
        if not os.path.isfile(behavioral_info):
            print('You miss the file ./data/behavioral_info.xlsx. Quit!')
        else:
            h_test(behavioral_info)
            print('H-test is finished.')
    else:
        raise NotImplementedError('Unknown input for --analysis! Valid input is tANOVA, ttest or htest')

if __name__ == "__main__":
    opt, _ = Options().parse(if_print=False)
    analysis_type = opt.analysis
    dataset_path = opt.eeg_dataset
    splits_path = opt.splits_path
    fig_path = opt.fig_path
    save_mid_result = opt.save_results
    main(analysis_type, dataset_path, splits_path, fig_path, save_mid_result=save_mid_result)