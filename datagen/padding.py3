import numpy as np

DEFAULT_DTYPE = 'int32'
def get_pad_mask(size, starts, dtype=DEFAULT_DTYPE):
    '''
    size: A tuple of (batch, maxlen).
    starts: 
        The lower-bound indices where the padding begins.
        Or the numbers of element to skip before the padding begins.
        Value for each element should inclusively fall between [1, maxlen].
        Shape (batch,).

    >>> get_pad_mask([3,4], [1,2,4])
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    '''
    assert np.ndim(size) == 1
    assert np.shape(size)[0] == 2
    batch = size[0]
    maxlen = size[1]
    assert np.ndim(starts) == 1
    assert np.shape(starts)[0] == batch
    assert np.all(np.greater(starts, 0))
    assert np.all(np.less_equal(starts, maxlen))
    result = np.repeat([np.arange(maxlen)], batch, 0)
    result = result < np.expand_dims(starts, -1)
    result = result.astype(dtype)
    return result
def get_random_pad_mask(size, start, dtype=DEFAULT_DTYPE):
    '''
    size: A tuple of (batch, maxlen).
    start: 
        The lower-bound index where the padding may begin.
        Or the numbers of element to skip before the padding may begin.
        Value should inclusively fall between [1, maxlen].
    
    >>> get_random_pad_mask([3,4], 1)
    array([[1, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    '''
    assert np.ndim(size) == 1
    assert np.shape(size)[0] == 2
    batch = size[0]
    maxlen = size[1]
    assert np.ndim(start) == 0
    assert start > 0
    assert start <= maxlen
    starts = np.random.randint(start, maxlen + 1, size=batch)
    return get_pad_mask(size, starts)
