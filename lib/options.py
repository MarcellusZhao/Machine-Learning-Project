import argparse
import os
import sys

class Options():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--eeg_dataset', type=str, default='data/training_data/EEG_dataset.pth', help="EEG dataset path")
        self._parser.add_argument('--splits_path', type=str, default="data/training_data/splits.pth", help="splits path")
        self._parser.add_argument('--split_num', type=int, default=0, help="split number")

        # For training
        self._parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        self._parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
        self._parser.add_argument('--train_mode', type=str, default='full', help='training mode: full/window/channel')
        self._parser.add_argument('--classifier', type=str, default='CNN', help="GRU/MLP/CNN/LIN")
        self._parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
        self._parser.add_argument('--epochs', default=100, type=int, help='training epochs')
        self._parser.add_argument('--GPUindex', default=0, type=int, help='which GPU to use')
        self._parser.add_argument('--no_cuda', default=False, help="disable CUDA", action="store_true")
        self._parser.add_argument('--window_s', default=0, type=int, help='the starting point of the window')
        self._parser.add_argument('--channel_idx', default=0, type=int, help='the idx of the channel') 
        self._parser.add_argument('--save_path', type=str, default='checkpoints', help='the path to save trained models')
        self._parser.add_argument('--save_model', default=False, help="save checkpoints", action="store_true")

        # For evaluation
        self._parser.add_argument('--load_path', type=str, default="checkpoints", help="path to load trained model")

        # For Guided Backprop
        self._parser.add_argument('--gif', default=False, help="generate gif", action="store_true")

        # For statistic analysis
        self._parser.add_argument('--analysis', type=str, default='', help="type of analysis: tANOVA/t-test/h-test")
        self._parser.add_argument('--fig_path', type=str, default='./figs', help="path to save generated figures")
        self._parser.add_argument('--save_results', default=False, help="save middle results", action="store_true")
        
        self._initialized = True

    def parse(self, if_print=True):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        self.set_train_mode_param()

        if not os.path.exists(self._opt.save_path):
            try:
                os.makedirs(self._opt.save_path)
            except:
                print('Invalid path to save models! Quit!')
                sys.exit()

        # get and set gpus
        args = vars(self._opt)

        # print in terminal args
        if if_print:
            self._print(args)

        return self._opt, self.train_mode_param[self._opt.train_mode]

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def set_train_mode_param(self):
    
        self.train_mode_param = {'full':{'channel_idx':None,
                                         'channel_num':128, 
                                         'eeg_length':300},
                                         
                                 'window':{'channel_idx':None,
                                           'channel_num':128, 
                                           'eeg_length':80},

                                 'channel':{'channel_idx':self._opt.channel_idx,
                                           'channel_num':1, 
                                           'eeg_length':300},}
