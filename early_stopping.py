import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation acc doesn't improve after a given patience."""

    def __init__(self, seed, feature, label_name, loop, save_path, patience=15, verbose=False, delta=1e-5):
        """
        Args:
            save_path : tha path to save model
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.seed = seed
        self.feature = feature
        self.label_name = label_name
        self.loop = loop
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation acc decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, str(self.seed)+'_'+str(self.feature)+'_'+str(self.label_name)+'_'+str(self.loop) + '.pth')
        torch.save(model, path)
        self.val_acc_max = val_acc

