# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview/evaluation
# https://www.kaggle.com/thariqnugrohotomo/metric-map-k-4

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

def mapk(
    ytrue, ypred, k,
    return_sequences=0, write=False, verbose=0,
    skip_empty_ytrue=1, ndigit=None, validate=1,
    n_jobs=1, batch=99999,
):
    '''
    MAP@K = Mean Average Precision @ K
    
    `ypred` must be a square array.
    `ytrue` can be ragged/jagged.
    Both of them can be a '`list` of `list`' or '`dict` of `list`'.
    '''
    
    assert skip_empty_ytrue, 'not implemented yet.'
    if hasattr(ytrue,'to_dict'):
        ytrue = ytrue.to_dict()
    if hasattr(ypred,'to_dict'):
        ypred = ypred.to_dict()
    assert isinstance(ytrue,dict)==isinstance(ypred,dict), (type(ytrue),type(ypred))
    if isinstance(ytrue,dict) and isinstance(ypred,dict):
        true = set(ytrue.keys())
        assert len(true)==len(ytrue), 'duplicate ytrue'
        pred = set(ypred.keys())
        assert len(pred)==len(ypred), 'duplicate ypred'
        dt = true-pred
        assert not dt, f'missing prediction for {list(dt)[:5]}'
        keys = ytrue.keys()
        ytrue = list(ytrue.values())
        assert all(map(len,ytrue)), 'found an empty ytrue'
        ypred = [ypred[kk] for kk in keys]
    n = len(ytrue)
    assert n, n
    assert n==len(ypred), (n,len(ypred))
    assert k>0, k
    ypred = np.array(ypred)
    assert ypred.shape[1] == k, ypred.shape
    if validate:
        tmp = pd.Series(ypred.flatten())
        tmp = tmp.groupby(np.repeat(np.arange(len(ypred)), k), sort=False)
        tmp = tmp.nunique()
        assert (tmp>=k).all(), f'found duplicate item in ypred ({tmp[tmp<k].index[0]}) {ypred[tmp[tmp<k].index[0]]}'
    def calc(arange):
        scores = []
        for ii in arange:
            true = ytrue[ii]
            if skip_empty_ytrue and len(true)==0:
                continue
            pred = ypred[ii]
            ans = 0
            isec = np.intersect1d(pred, true, return_indices=True)[1]
            if len(isec):
                isec.sort()
                hi = np.arange(len(isec))+1
                lo = isec+1
                ans = (hi/lo).sum() / min(k, len(true))
            scores.append(ans)
        return scores
    arange = np.arange(n)
    if n_jobs==1:
        if verbose:
            arange = tqdm(arange)
        scores = calc(arange)
    else:
        print('WARNING: parallel mode is not tested yet.')
        split = np.array_split(arange, n//batch)
        scores = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
        )(delayed(calc)(ii) for ii in split)
        scores = np.hstack(scores)
    assert len(scores), 'ytrue elements are blanks?'
    if return_sequences:
        assert keys, '.keys()'
        assert len(keys)==len(scores), (len(keys),len(scores))
        return pd.Series(scores,index=keys).round(3)
    ans = np.mean(scores)
    if ndigit is not None:
        ans = round(ans,ndigit)
    if write:
        s = ans
        if ndigit is not None:
            s = str(s).ljust(ndigit+2,'0')
        print(f'>>[ {s} ]<< customers:{len(ytrue)}')
    return ans
