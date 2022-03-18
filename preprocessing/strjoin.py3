def strjoin(*series, sep=' '):
    '''
    Joining a sequence of `series` separating them with `sep`.
    Analogous to `sep.join(series)`.
    
    `series` is pandas.Series
    
    Example:
    >>> strjoin(df.firstname, df.middlename, df.lastname)
    >>> strjoin(df.drive, df.directory, df.filename, sep='/')
    '''
    ans = series[0].astype('string')
    for ii in range(1, len(series)):
        ans += sep
        ans += series[ii].astype('string')
    return ans
