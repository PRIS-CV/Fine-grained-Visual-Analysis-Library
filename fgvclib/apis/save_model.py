import torch
import os
from fgvclib.utils.logger.base_logger import BaseLogger

def save_model(cfg, model, logger: BaseLogger):
    if cfg.WEIGHT.NAME:
        
        if not os.path.exists(cfg.WEIGHT.SAVE_DIR):
            try:
                os.mkdir(cfg.WEIGHT.SAVE_DIR)
            except:
                logger.add_log_item(f'Cannot create save dir under {cfg.WEIGHT.SAVE_DIR}')
                logger.close()
                exit()
        save_path = os.path.join(cfg.WEIGHT.SAVE_DIR, cfg.WEIGHT.NAME)
        torch.save(model.state_dict(), save_path)
        logger.add_log_item(f'Saving checkpoint to {save_path}')