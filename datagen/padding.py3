import numpy as np

DEFAULT_DTYPE = int
def get_padding_mask(size, lengths, dtype=DEFAULT_DTYPE):
    '''
    >>> get_padding_mask([3,5], [1,2,5])
    array([[1, 0, 0, 0, 0],
           [1, 1, 0, 0, 0],
           [1, 1, 1, 1, 1]])
    '''
    assert np.ndim(size) == 1
    assert np.shape(size)[0] == 2
    height = size[0]
    max_width = size[1]
    assert np.ndim(lengths) == 1
    assert np.shape(lengths)[0] == height
    assert np.all(np.greater(lengths, 0))
    assert np.all(np.less_equal(lengths, max_width))
    result = np.repeat([np.arange(max_width)], height, 0)
    result = result < np.expand_dims(lengths, -1)
    result = result.astype(dtype)
    return result
def get_random_padding_mask(size, min_length, dtype=DEFAULT_DTYPE):
    '''
    >>> get_random_padding_mask([3,5], 1)
    array([[1, 1, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0]])
    
    >>> get_random_padding_mask([3,5], 5)
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]])
    '''
    assert np.ndim(size) == 1
    assert np.shape(size)[0] == 2
    height = size[0]
    max_width = size[1]
    assert np.ndim(min_length) == 0
    assert min_length > 0
    assert min_length <= max_width
    lengths = np.random.randint(min_length, max_width + 1, size=height)
    return get_padding_mask(size, lengths, dtype)

def append_before_padding(matrix, val, padval):
    '''
    >>> mat = [[1, 2, 0, 0], [1, 2, 3, 4]]
    >>> append_before_padding(mat, 9, 0)
    array([[1, 2, 9, 0, 0],
           [1, 2, 3, 4, 9]])
    '''
    assert np.ndim(matrix) == 2
    assert np.ndim(padval) == 0
    result = matrix
    result = np.concatenate([matrix, np.expand_dims([padval] * np.shape(matrix)[0], -1)], -1)
    np.put_along_axis(
        result,
        np.argmax(result == padval, -1, keepdims=True),
        val,
        -1,
    )
    return result

