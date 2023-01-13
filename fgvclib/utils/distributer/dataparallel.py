import torch.nn as nn


class FGVCDataParallel(nn.DataParallel):

    def __getattr__(self, name):
        return getattr(self.module, name)
