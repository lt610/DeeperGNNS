import torch as th
import datetime
import os


class EarlyStoppingBoth(object):
    def __init__(self):
        self.best_acc = None
        self.best_loss = None
        self.acc_path = "../result/model_state/acc.pt"
        self.loss_path = "../result/model_state/loss.pt"
        self.is_stop = False

    def __call__(self, acc, loss, model):

        if self.best_acc is None:
            self.best_acc = acc

        if self.best_loss is None:
            self.best_loss = loss

        if self.best_acc < acc:
            self.best_acc = acc
            self.save_acc_checkpoint()
        if self.best_loss > loss:
            self.best_loss = loss
            self.save_loss_checkpoint()

    def save_acc_checkpoint(self, model):
        th.save(model.state_dict(), self.acc_path)

    def load_acc_checkpoint(self):
        return th.load(self.acc_path)

    def save_loss_checkpoint(self, model):
        th.save(model.state_dict(), self.loss_path)

    def load_loss_checkpoint(self):
        return th.load(self.loss_path)