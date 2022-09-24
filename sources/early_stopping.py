import numpy as np
import torch
import logging
from utils import init_logger
class Early_Stopping:
    def __init__(self, patience=50, verbose=False, delta=0):
        init_logger()
        self.logger = logging.getLogger(__name__)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'Loss stagnation counter: {self.counter}\n')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,path):
        '''loss가 감소하면 모델 저장'''
        if self.verbose:
            self.logger.info(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})\n')
            self.logger.info("Saving model checkpoint to {}\n".format(path))
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class Early_Stopping_multi:
    def __init__(self, patience=50, verbose=False, delta=0):
        init_logger()
        self.logger = logging.getLogger(__name__)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'Loss stagnation counter: {self.counter}\n')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,path):
        if self.verbose:
            self.logger.info(f'Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})\n')
            self.logger.info("Saving model checkpoint to {}\n".format(path))
        torch.save(model.module.state_dict(), path)
        self.val_loss_min = val_loss

class Early_Stopping_F1:
    def __init__(self, patience=50, verbose=False, delta=0):
        init_logger()
        self.logger = logging.getLogger(__name__)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path,train_f1):

        score = train_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_f1, model, path,train_f1)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'F1 score stagnation counter: {self.counter}\n')
        else:
            self.best_score = score
            self.save_checkpoint(train_f1, model, path,train_f1)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,path,train_f1):
        if self.verbose:
            self.logger.info(f'Loss decreased ({self.val_loss_min:.6f} --> {train_f1:.6f})\n')
            self.logger.info("Saving model checkpoint to {}\n".format(path))
        torch.save(model.state_dict(), path)
        self.val_loss_min = train_f1