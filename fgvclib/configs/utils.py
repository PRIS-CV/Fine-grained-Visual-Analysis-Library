def turn_list_to_dict(arg_list:list):
    r"""Turn a list of arguments into a dict.

        Args:
            arg_list (list): The arguments list.

    """
    if arg_list:
        d = dict()
        for arg in arg_list:
            d.update(arg)
        return d
    else:
        return None
