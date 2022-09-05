import wandb
import typing as t
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        wandb.init(project=cfg.EXP_NAME)
        wandb.config = self.convert_to_dict(self.cfg)
        self.loss_values = dict()
        
    def record_loss(self, losses: t.Dict) -> None:
        wandb.log(losses)

    def record_eval_res(self, res: t.Dict) -> None:
        wandb.log(res)

    def plot_line(self, x_values: t.Sequence, y_values: t.Sequence, graph_name: str):
        data = [[x, y] for (x, y) in zip(x_values, y_values)]
        table = wandb.Table(data=data, columns = ["x", "y"])
        wandb.log({graph_name : wandb.plot.line(table, "x", "y", title=graph_name)})

    def close(self):
        wandb.config.update({
            "End-Time": self.get_time_point()
        })
    

