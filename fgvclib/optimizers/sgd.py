from torch.optim import SGD


from .import optimizer

@optimizer("SGD")
def sgd(params, lr, cfg):
    return SGD(params=params, lr=lr, **cfg)