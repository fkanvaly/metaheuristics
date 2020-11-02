import numpy as np

__all__ = [
        'x', 'y', 'distance',
       'algo',
       'make',
       'iters',
       'num',
       'bit',
       'plot',
       'pb',
   ]

########################################################################
# Utilities
########################################################################

def x(a):
    """Return the first element of a 2-tuple.
    >>> x([1,2])
    1
    """
    return a[0]


def y(a):
    """Return the second element of a 2-tuple.
    >>> y([1,2])
    2
    """
    return a[1]


def distance(a,b):
    """Euclidean distance (in pixels).

    >>> distance( (1,1),(2,2) ) == math.sqrt(2)
    True
    """
    return np.sqrt( (x(a)-x(b))**2 + (y(a)-y(b))**2 )

