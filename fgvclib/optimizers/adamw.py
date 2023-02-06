from torch.optim import AdamW


from .import optimizer

@optimizer("AdamW")
def adamw(params, lr, cfg):
    return AdamW(params=params, lr=lr, **cfg)