import numpy as np

DEFAULT_DTYPE = 'int32'
def get_pad_mask(size, skips, dtype=DEFAULT_DTYPE):
    '''
    size: A tuple of (batch, maxlen).
    skips: 
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
    assert np.ndim(skips) == 1
    assert np.shape(skips)[0] == batch
    assert np.all(np.greater(skips, 0))
    assert np.all(np.less_equal(skips, maxlen))
    result = np.repeat([np.arange(maxlen)], batch, 0)
    result = result < np.expand_dims(skips, -1)
    result = result.astype(dtype)
    return result
def get_random_pad_mask(size, skip, dtype=DEFAULT_DTYPE):
    '''
    size: A tuple of (batch, maxlen).
    skip: 
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
    assert np.ndim(skip) == 0
    assert skip > 0
    assert skip <= maxlen
    skips = np.random.randint(maxlen, size=batch) + 1
    return get_pad_mask(size, skips)
