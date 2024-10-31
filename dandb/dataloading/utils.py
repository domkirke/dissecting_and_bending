import typing as tp 
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Subset

from .simple_dataset import *


def make_loader(dataset_path: str, 
                 bs: int, 
                 num_workers: int = 0, 
                 max_samples: tp.Optional[int] = None
                 )-> torch.utils.data.DataLoader:
    dataset = SimpleDataset(
        dataset_path, 
    )    
    @torch.no_grad()
    def collate_fn(B):
        B_wav = [torch.tensor(x).float()for x in B]
        B_wav = pad_sequence(B_wav, batch_first=True, padding_value=0)
        if max_samples is not None and B_wav.shape[-1]>max_samples:
            B_wav = B_wav[..., :max_samples]
        return B_wav
        
    audio_loader = DataLoader(
        dataset, 
        batch_size=bs, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        shuffle=True
        )
    return audio_loader