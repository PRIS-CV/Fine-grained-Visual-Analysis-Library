import wandb
import typing as t
from .logger import Logger


class WandbLogger(Logger):

    def __init__(self, exp_name, project, path) -> None:
        super().__init__(exp_name)
        wandb.init(project=project, name=self.exp_name, dir=path)
        wandb.config = self.convert_to_dict(self.cfg)

    def __call__(self, item):
        return super().__call__(item)

    def _record(self, item: t.Union[dict, str], step:t.Optional[int]=0, acc:t.Optional[bool]=False):
        if isinstance(item, dict):
            wandb.log(item)
            info = self._sum_info(item, acc)
        else:
            info = item
            wandb.config.update({
                step: info
            })

    def finish(self):
        wandb.config.update({
            "End-Time": self.get_time_point()
        })
        wandb.finish()

def wandb_logger(cfg) -> Logger:
    return WandbLogger(exp_name=cfg.EXP_NAME, dir=cfg.LOGGER.FILE_PATH, project=cfg.LOGGER.PROJ_NAME)
    

