# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_mini_batch_training.ipynb.

# %% auto 0
__all__ = ['accuracy', 'report', 'Dataset', 'fit', 'get_dls']

# %% ../nbs/04_mini_batch_training.ipynb 1
import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import gzip, pickle, matplotlib.pyplot as plt

# %% ../nbs/04_mini_batch_training.ipynb 29
def accuracy(out, yb):
    return (out.argmax(1)==yb).float().mean()

# %% ../nbs/04_mini_batch_training.ipynb 34
def report(loss, preds, yb):
    print(f"loss: {loss:.2f}, accuracy: {accuracy(preds, yb):.2f}")

# %% ../nbs/04_mini_batch_training.ipynb 75
class Dataset:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

# %% ../nbs/04_mini_batch_training.ipynb 109
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler

# %% ../nbs/04_mini_batch_training.ipynb 120
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_func(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            total_loss, total_acc, count = 0., 0., 0
            for xb, yb in valid_dl:
                preds = model(xb)
                n = len(xb)
                count += n
                total_loss += loss_func(preds, yb).item()*n
                total_acc += accuracy(preds, yb).item()*n
        print(epoch, total_loss/count, total_acc/count)
    
    return total_loss/count, total_acc/count           

# %% ../nbs/04_mini_batch_training.ipynb 121
def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, bs*2, shuffle=False, **kwargs)
    )