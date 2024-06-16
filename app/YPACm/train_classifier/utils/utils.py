# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Iterable
import cv2

# files IO
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def setify(x):
    # print(f'type {type(x)}, listify {listify(x)}')
    return x if isinstance(x, set) else set(listify(x))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p / f for f in fs if f is not f.startswith('.')
           and ((not extensions) or f".{f.split('.')[-1].lower()}" in extensions)]  # check for sure folders
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    """get all files path"""
    path = Path(path)
    extensions = setify(extensions) if isinstance(extensions, str) \
        else setify(e.lower() for e in extensions)

    if recurse:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path)):
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith(('.', '__'))]
            res += _get_files(p, f, extensions)
        return sorted(res)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return sorted(_get_files(path, f, extensions))


def show_tensor(tensor):
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    cv2.namedWindow('display', cv2.WND_PROP_FULLSCREEN)
    try:
        cv2.imshow('display', np_img[..., ::-1])
        cv2.waitKey(0)
    except Exception as e:
        print(f'ERROR {e}')
    finally:
        cv2.destroyAllWindows()
