from torch.optim import Adam


from .import optimizer

@optimizer("Adam")
def adam(params, lr, cfg):
    return Adam(params=params, lr=lr, **cfg)

