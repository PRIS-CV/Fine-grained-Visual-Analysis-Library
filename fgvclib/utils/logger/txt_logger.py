import os
import typing as t

from .base_logger import BaseLogger


class TxtLogger(BaseLogger):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.file_path = self.cfg.LOGGER.FILE_PATH
        self.log_str = [self.convert_to_jsonstr(self.convert_to_dict(self.cfg)), "Start Time: " + self.start_time_point]
        self.file_name = os.path.join(self.cfg.LOGGER.FILE_PATH, self.cfg.EXP_NAME + self.start_time_point + ".txt")
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

    def record_loss(self, losses: t.Dict) -> None:
        self.add_log_item(self.convert_to_jsonstr(losses))
    
    def record_eval_res(self, res: t.Dict) -> None:
        self.add_log_item(self.convert_to_jsonstr(res))

    def add_log_item(self, item: str) -> None:
        self.log_str.append(item)
    
    def write_log_file(self) -> None:
        with open(self.file_name,'w') as f:
            for item in self.log_str:
                f.write(item + "\n")
            f.close()
    
    def close(self):
        self.end_time_point = self.get_time_point()
        self.add_log_item("End Time: " + self.end_time_point)
        self.add_log_item("Finish")
        self.write_log_file()
    