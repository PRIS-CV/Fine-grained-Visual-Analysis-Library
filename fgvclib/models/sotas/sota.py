import torch.nn as nn


class FGVCSOTA(nn.Module):
    
    def __init__(self, backbone:nn.Module, encoding:nn.Module, necks:nn.Module, heads:nn.Module, criterions:nn.Module):
        super(FGVCSOTA, self).__init__()

        self.backbone = backbone
        self.necks = necks
        self.encoding = encoding
        self.heads = heads
        self.criterions = criterions

    def get_structure(self):
        ss = f"\n{'=' * 30}\n" + f"\nThe Structure of {self.__class__.__name__}:\n" + f"\n{'=' * 30}\n\n"
        for s in ['backbone', 'encoding', 'necks', 'heads']:
            m = getattr(self, s)
            ss += f"{s}: {m.__class__.__name__}\n"
        ss += f"\n{'=' * 30}\n"
        return ss
        