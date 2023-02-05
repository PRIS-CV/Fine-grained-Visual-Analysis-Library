from torch.optim import Adam


from .import optimizer

@optimizer("Adam")
def adam(params, cfg):
    return Adam(params=params, **cfg)

