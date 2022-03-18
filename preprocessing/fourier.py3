import matplotlib.pyplot as plt
import numpy as np

def fourier(df, column, lo, nunique, prefix=None, plot=True):
    '''
    Generate fourier/trigonometric feature for/from a periodic/cyclic discrete values.
    
    `df` is pandas.DataFrame where the generated features will be added as columns.
    
    `columns` can be a `pd.Series`, `np.array`, or column name in `df`.
    It's contain periodic/cyclic discrete values where the feature will be generated from.
    
    `lo` is the lowest possible value (inclusive) of the periodic/cyclic discrete values.
    `nunique` is the cardinality of the periodic/cyclic discrete values.
    Examples:
    - Minutes of hour : `lo=0, nunique=60`.
    - Hours of day : `lo=0, nunique=24`.
    - Months of year : `lo=1, nunique=12` (January is usually denoted as `1`).
    
    The generated columns name will be prefixed with `prefix`.
    If `prefix` is not defined, it will be inferred from `column` whenever possible.
    
    `plot` whether to visualize the generated feature to aid debugging.
    '''
    
    series = column
    if isinstance(series, str):
        series = df[series]
    if prefix is None:
        assert hasattr(series, 'name'), "`prefix` can't be None."
        prefix = series.name
    norm = (series-lo) / nunique
    sin = np.sin(norm*2*np.pi)
    cos = np.cos(norm*2*np.pi)
    df[f'{prefix}sin'] = sin
    df[f'{prefix}cos'] = cos
    if plot:
        # sinusoidal wave
        plt.figure(figsize=(4,2))
        plt.scatter(series, sin)
        plt.scatter(series, cos)
        plt.title(prefix)
        plt.show()
        # circle / oval
        plt.figure(figsize=(4,2))
        plt.scatter(sin, cos)
        plt.title(prefix)
        plt.show()
