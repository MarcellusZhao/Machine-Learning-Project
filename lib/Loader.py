import torch

class EEGDataset:
    def __init__(self, eeg_signals_path, label_tag="genders", use_window=False, window_len=100, window_s=0):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"].type(torch.FloatTensor)
        self.label_tag = label_tag
        self.labels = loaded[self.label_tag].type(torch.LongTensor)

        # mean/std computed from training set
        # different for different split
        self.means = loaded["means"][0]
        self.stddevs = loaded["stddevs"][0]

        self.use_window = use_window
        self.window_len = window_len
        self.window_s = window_s

        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # normalize EEG
        eeg = (self.data[i] - self.means)/self.stddevs # (128, 300)
        if self.use_window:
            eeg = eeg[:, self.window_s: self.window_s+self.window_len]

        eeg = eeg.t()

        # Get label
        label = self.labels[i]

        return eeg, label

class EEGDataset_window:
    def __init__(self, eeg_signals_path, idx, label_tag="genders", use_window=False, window_len=100, window_s=0):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"].type(torch.FloatTensor)
        self.label_tag = label_tag
        self.labels = loaded[self.label_tag].type(torch.LongTensor)

        self.use_window = use_window
        self.window_len = window_len
        self.window_s = window_s

        # mean/std computed from training set
        # different for different split
        self.means = torch.mean(self.data[idx, :, self.window_s: self.window_s+self.window_len], dim=(0,2), keepdim=True)[0]  #loaded["means"][0]
        self.stddevs = torch.std(self.data[idx, :, self.window_s: self.window_s+self.window_len], unbiased=False, dim=(0,2), keepdim=True)[0] #loaded["stddevs"][0]
        #print(self.means.shape)
        #print(self.stddevs.shape)

        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # normalize EEG
        eeg = (self.data[i] - self.means)/self.stddevs # (128, 300)
        if self.use_window:
            eeg = eeg[:, self.window_s: self.window_s+self.window_len]

        eeg = eeg.t()

        # Get label
        label = self.labels[i]

        return eeg, label

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Compute size
        self.size = len(self.split_idx)



    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label