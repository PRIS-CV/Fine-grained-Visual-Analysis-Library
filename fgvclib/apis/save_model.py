import torch
import torch.nn as nn
import os
from yacs.config import CfgNode

from fgvclib.utils.logger.logger import Logger



def save_model(cfg: CfgNode, model: nn.Module, logger: Logger):
    r"""Save the trained FGVC model.

    Args:
        cfg (CfgNode): 
            The root config node.
        model (nn.Module): 
            The FGVC model.
        logger (Logger): 
            The Logger object.
    """
    
    if cfg.WEIGHT.NAME:
        
        if not os.path.exists(cfg.WEIGHT.SAVE_DIR):
            try:
                os.mkdir(cfg.WEIGHT.SAVE_DIR)
            except:
                logger(f'Cannot create save dir under {cfg.WEIGHT.SAVE_DIR}')
                logger.finish()
                exit()
        save_path = os.path.join(cfg.WEIGHT.SAVE_DIR, cfg.WEIGHT.NAME)
        torch.save(model.state_dict(), save_path)
        logger(f'Saving checkpoint to {save_path}')