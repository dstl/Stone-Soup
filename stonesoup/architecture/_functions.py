from string import ascii_uppercase as auc


def _dict_set(my_dict, value, key1, key2=None):
    """Utility function to add value to my_dict at the specified key(s)
    Returns True if the set increased in size, i.e. the value was new to its position"""
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


def _default_label(node, last_letters):
    """Utility function to generate default labels for nodes, where none are given
    Takes a node, and a dictionary with the letters last used for each class,
    ie `last_letters['Node']` might return 'AA', meaning the last Node was labelled 'Node AA'"""
    node_type = type(node).__name__
    type_letters = last_letters[node_type]  # eg 'A', or 'AA', or 'ABZ'
    new_letters = _default_letters(type_letters)
    last_letters[node_type] = new_letters
    return node_type + ' ' + new_letters, last_letters


def _default_letters(type_letters) -> str:
    if type_letters == '':
        return 'A'
    count = 0
    letters_list = [*type_letters]
    # Move through string from right to left and shift any Z's up to A's
    while letters_list[-1 - count] == 'Z':
        letters_list[-1 - count] = 'A'
        count += 1
        if count == len(letters_list):
            return 'A' * (count + 1)
    # Shift current letter up by one
    current_letter = letters_list[-1 - count]
    letters_list[-1 - count] = auc[auc.index(current_letter) + 1]
    new_letters = ''.join(letters_list)
    return new_letters
