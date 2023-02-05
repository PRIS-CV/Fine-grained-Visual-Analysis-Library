from torch.optim import AdamW


from .import optimizer

@optimizer("AdamW")
def adamw(params, cfg):
    return AdamW(params=params, **cfg)