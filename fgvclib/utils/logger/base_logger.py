import typing as t
import json
import time

class BaseLogger(object):
    
    def __init__(self, cfg) -> None:

        self.cfg = cfg
        self.start_time_point = self.get_time_point()
        self.end_time_point = ""

    def record_loss(self, losses: t.Dict) -> None:
        raise NotImplementedError
    
    def record_eval_res(self, res: t.Dict) -> None:
        raise NotImplementedError
    
    def add_log_item(self, item: str) -> None:
        pass

    def convert_to_dict(self, cfg_node):
        cfg_dict = dict()
        for k, v in cfg_node.items():
            cfg_dict.update({
                k: v
            })
        return cfg_dict

    def convert_to_jsonstr(self, cfg_dict: t.Dict) -> str:
        return json.dumps(cfg_dict)

    def get_time_point(self):
        return time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))

    def close(self):
        self.end_time_point = self.get_time_point()

