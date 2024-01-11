from typing import Optional

import pickle


def pickle_load(path: Optional[str], verbose=False):
    if path is None:
        return None
    if verbose:
        print("load %s..." % path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
