from config import dataset_root
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pdb
import os
import argparse
# from models.backbones import create_backbone



def A():
    raise IndexError()

class D(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.samples = torch.range(0, 10)
    
    def __getitem__(self, idx):
        # A()
        raise IndexError()
        return self.samples[idx]
    def __len__(self):
        return len(self.samples)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', action='append', type=str, nargs='+', default='a')
    parser.add_argument('--b', type=int, nargs='+', default=0)

    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.arange(6).reshape(2,3).float()
    y = torch.rand(size=(2,3))
    print(y.shape)
    w1 = torch.nn.Linear(3,3,bias=False)
    w2 = torch.nn.Linear(3,3,bias=False)
    yh = w1(a)+w2(a)
    loss = torch.sqrt(torch.sum((y-yh)**2, 1)).mean()
    # loss += torch.sqrt(torch.sum((y-yh)**2, 1)).mean()
    loss = loss + torch.sqrt(torch.sum((y-yh)**2, 1)).mean()
    loss.backward()
    print(w1.weight.grad)