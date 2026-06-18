from __future__ import print_function, division

import numpy as np
import torch
from torch.utils.data import Dataset


class MOF_ID_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.mofid = self.data[:, 0].astype(str)

        self.tokens = np.array([
            tokenizer.encode(i, max_length=512, truncation=True, padding='max_length')
            for i in self.mofid
        ])
        self.label = self.data[:, 1].astype(float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index])).long()
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1).float()
        return X, y
