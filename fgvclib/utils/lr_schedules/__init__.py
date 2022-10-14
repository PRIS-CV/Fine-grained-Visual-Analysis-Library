from .cosine_anneal_schedule import cosine_anneal_schedule

__all__ = [
    'cosine_anneal_schedule'
]


def get_lr_schedule(lr_schedule_name):
    r"""Return the learning rate schedule with the given name.

        Args: 
            lr_schedule_name (str): 
                The name of learning rate schedule.
        
        Return: 
            The learning rate schedule contructor method.
    """

    if lr_schedule_name not in globals():
        raise NotImplementedError(f"Learning rate schedule not found: {lr_schedule_name}\nAvailable learning rate schedules: {__all__}")
    return globals()[lr_schedule_name]
