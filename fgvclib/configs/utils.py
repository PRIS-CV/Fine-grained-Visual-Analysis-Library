def turn_list_to_dict(arg_list):
    if arg_list:
        d = dict()
        for arg in arg_list:
            d.update(arg)
        return d
    else:
        return None
