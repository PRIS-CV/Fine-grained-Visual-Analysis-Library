import os
import typing as t
import time

from .logger import Logger


class TxtLogger(Logger):
    
    def __init__(self, exp_name:str, path:str, show_frequence:t.Optional[int]=50):
        r"""The text logger for record loss and other data.
            Args:
                exp_name (str): 
                    The experiment name used to named the record file.
                path (str): 
                    The file directory used to store the log files.
                show_freqence (str): 
                    Print the data per n steps.
        """
        start_point = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        super(TxtLogger, self).__init__(exp_name + "_" + start_point)
        if not os.path.exists(path):
            os.mkdir(path)
        print(f"Experiment log recorded in {path}.")
        self.path = path
        self.buffer = ""
        assert show_frequence >= 0, 'The logger\'s print freqence should larger than or equal 0'
        self.show_frequence = show_frequence 

        with open(os.path.join(self.path, self.exp_name + ".txt"), 'w') as f:
            f.write("Start: " + start_point + "\n")
            f.close()
    
    def __call__(self, item: t.Union[dict, str], step:t.Optional[int]=0, acc:t.Optional[bool]=False):
        return self._record(item, step, acc)

    def _record(self, item: t.Union[dict, str], step:t.Optional[int]=0, acc:t.Optional[bool]=False):
        if isinstance(item, dict):
            info = self._sum_info(item, acc)
        else:
            info = item
        if step % self.show_frequence == 0:
            self.buffer += info + "\n"
            self.write_to_file()

    def write_to_file(self):
        with open(os.path.join(self.path, self.exp_name + ".txt"), 'a') as f:
            f.write(self.buffer)
            f.close()
        self._clear_buffer()
        
    def _add_buffer(self, info: str):
        self.buffer += info + "\n"

    def _clear_buffer(self):
        self.buffer = ""

    def _sum_info(self, item: dict, acc=False):
        info = ""
        for k, v in item.items():
            info += k
            info += ": "
            info += f"{v:.2f} " if isinstance(v, float) and not acc else f"{v} "
        return info

def txt_logger(cfg) -> Logger:
    return TxtLogger(cfg.EXP_NAME, path=cfg.LOGGER.FILE_PATH, show_frequence=cfg.LOGGER.PRINT_FRE)
