from timm.data import Mixup       

class MixUpCutMix:
    def __init__(self, cfg: dict, num_classes:int):
        self.mixup_fn = Mixup(
            mixup_alpha=cfg['mixup_alpha'], cutmix_alpha=cfg['cutmix_alpha'], cutmix_minmax=cfg['cutmix_minmax'],
            prob=cfg['prob'], switch_prob=cfg['switch_prob'], mode=cfg['mode'],
            label_smoothing=cfg['label_smoothing'], num_classes=num_classes)
    
    def __call__(self, samples, targets):
        samples, targets = self.mixup_fn(samples, targets)
        return samples, targets 