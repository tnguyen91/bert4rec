import torch
from torch.utils.data import Dataset
import random

class BERT4RecDataset(Dataset):
    def __init__(self, user_sequences, max_seq_len, mask_prob=0.15, seed=42):
        self.user_sequences = user_sequences
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        self.pad_token = 0
        self.mask_token = 1

        random.seed(seed)

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        seq = self.user_sequences[idx]
        seq = seq[-self.max_seq_len:]
        padding = [self.pad_token] * (self.max_seq_len - len(seq))
        seq = padding + seq

        masked_seq = []
        labels = []
        for item in seq:
            if item != self.pad_token and random.random() < self.mask_prob:
                masked_seq.append(self.mask_token)
                labels.append(item)
            else:
                masked_seq.append(item)
                labels.append(-100)

        return torch.LongTensor(masked_seq), torch.LongTensor(labels)
