class LRSchedule:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def step(**kwargs):
        raise NotImplementedError("Eacbh subclass of LRSchedule should implemented the step method.")
