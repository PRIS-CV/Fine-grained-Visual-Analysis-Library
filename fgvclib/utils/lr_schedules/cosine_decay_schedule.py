import math
import numpy as np


def cosine_decay(max_epochs, warmup_batchs, max_lr, batchs: int, decay_type: int = 1):
    total_batchs = max_epochs * batchs
    iters = np.arange(total_batchs - warmup_batchs)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (max_lr - 1e-12) * (1 + math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = max_lr * np.array([math.cos(7 * math.pi * t / (16 * total_batchs)) for t in iters])
    else:
        raise ValueError("Not support this decay type")

    if warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, max_lr, warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule
    