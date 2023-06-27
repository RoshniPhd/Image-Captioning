import pickle
from os import path as path_


def load(path):
    with open(path + '.pkl', 'rb') as f:  # 'rb' for reading; can be omitted
        my_dict = pickle.load(f)  # load file content as my_dict
    return my_dict


def save(path, file, override=False):
    if not path_.exists(path + '.pkl'):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(file, f)

    elif override:
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(file, f)
    else:
        pass
