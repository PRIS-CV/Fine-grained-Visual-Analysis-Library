class Logger(object):

    def __init__(self, exp_name) -> None:
        self.exp_name = exp_name

    def __call__(self, item):
        self._record(item)

    def _record(self, item):
        raise NotImplementedError

    def finish():
        print("Finish!")

