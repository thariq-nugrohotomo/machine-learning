import tqdm as _tqdm_

def tqdm(lst, auto=1, *args, **kwargs):
    f = _tqdm_.auto.tqdm if auto else _tqdm_.tqdm
    return f(lst, *args, **kwargs)
