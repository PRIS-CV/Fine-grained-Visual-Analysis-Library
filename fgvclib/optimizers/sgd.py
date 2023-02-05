from torch.optim import SGD


from .import optimizer

@optimizer("SGD")
def adamw(params, cfg):
    return SGD(params=params, **cfg)