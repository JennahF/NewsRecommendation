import torch
from utils.dict import Dictionary
from torch.utils.data import Dataset
import random
import config
from torchvision import transforms

def title_to_max_len(title_vec):
    pad = 48 - len(title_vec)
    if pad != 0:
        title_vec = title_vec + [0] * pad
    return title_vec

class myTrainSet(Dataset):
    def __init__(self, userId, history, candidate, mask):
        print()
        self.userId = userId
        self.history = history
        self.candidate = candidate
        self.mask = mask

    def __getitem__(self, index):
        return {'userId' : self.userId[index], 'history' : self.history[index], 'candidate' : self.candidate[index], 'mask': self.mask}

    def __len__(self):
        return len(self.userId)


class myTestSet(Dataset):
    def __init__(self, userId, history, candidate, label, mask):
        self.userId = userId
        self.history = history
        self.candidate = candidate
        self.label = label


    def __getitem__(self, index):
        return {'userId' : self.userId[index], 'history' : self.history[index], 'candidate' : self.candidate[index], 'label' : self.label[index], 'mask': self.mask}

    
    def __len__(self):
        return len(self.userId)