# coding: utf-8

import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR




# noinspection PyUnresolvedReferences
class CustomScheduler(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super().__init__(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)


    def get_lr(self):
        self.step()

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if ( self.last_epoch == 0 ) or ( self.last_epoch % self.step_size != 0 ):
            return [ group['lr'] for group in self.optimizer.param_groups ]

        return [ group['lr'] * self.gamma for group in self.optimizer.param_groups ]


    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]





def main():
    pass
