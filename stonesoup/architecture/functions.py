def _dict_set(my_dict, value, key1, key2=None):
    """Utility function to add value to my_dict at the specified key(s)
    Returns True iff the set increased in size, ie the value was new to its position"""
    if not my_dict:
        if key2:
            my_dict = {key1: {key2: {value}}}
        else:
            my_dict = {key1: {value}}
    elif key2:
        if key1 in my_dict:
            if key2 in my_dict[key1]:
                old_len = len(my_dict[key1][key2])
                my_dict[key1][key2].add(value)
                return len(my_dict[key1][key2]) == old_len + 1, my_dict
            else:
                my_dict[key1][key2] = {value}
        else:
            my_dict[key1] = {key2: {value}}
    else:
        if key1 in my_dict:
            old_len = len(my_dict[key1])
            my_dict[key1].add(value)
            return len(my_dict[key1]) == old_len + 1, my_dict
        else:
            my_dict[key1] = {value}
    return True, my_dict
