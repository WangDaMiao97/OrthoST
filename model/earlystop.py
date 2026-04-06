# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/4 20:51
@Author  : Linjie Wang
@FileName: earlystop.py
@Software: PyCharm
'''
import numpy as np
import torch
import os

class EarlyStop:
    """
    Early stops the training if loss doesn't improve after a given patience.

    Parameters
    ----------
    save_path : str
        Path to save the best model checkpoint.
    patience : int, default=7
        Number of epochs to wait before stopping after no improvement.
    delta : float, default=0.0
        Minimum change to qualify as an improvement.
    """
    def __init__(self, save_path, patience=7, delta=0.001):
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss, model):
        if self.best_score is None:
            self.best_score = loss
            self.save_checkpoint(loss, model)
        elif loss > (self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when validation loss decrease.'''
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        self.loss_min = loss