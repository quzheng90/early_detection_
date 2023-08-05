import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Potantial_User(Dataset):
    def __init__(self, dataset):
        ##text, mask, label, domain_label
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        #self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.domain_label = torch.from_numpy(np.array(dataset['event_label']))
        ###delete
        print('TEXT: %d, labe: %d, DOMAIN: %d' % (len(self.text), len(self.label), len(self.domain_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.mask[idx]), self.label[idx], self.domain_label[idx]