class LRSchedule:

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError("Eacbh subclass of LRSchedule should implemented the step method.")
