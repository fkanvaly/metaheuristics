from . import distance

########################################################################
# Objective functions
########################################################################

def coverage(domain, sensors, sensor_range):
    """Set a given domain's cells to on if they are visible
    from one of the given sensors at the given sensor_range.

    >>> coverage(np.zeros((5,5)),[(2,2)],2)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """
    for py in range(len(domain)):
        for px in range(len(domain[py])):
            p = (px,py)
            for x in sensors:
                if distance(x,p) < sensor_range:
                    domain[py][px] = 1
    return domain


def line(x0, y0, x1, y1):
    """Compute the set of pixels (integer coordinates) of the line
    between the given line (x0,y0) -> (x1,y1).
    Use the Bresenham's algorithm.
    This make a generator that yield the start and the end points.
    """
    dx = x1 - x0
    dy = y1 - y0

    if dx > 0:
        xs = 1
    else:
        xs = -1

    if dy > 0:
        ys = 1
    else:
        xs = -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        ax, xy, yx, ay = xs, 0, 0, ys
    else:
        dx, dy = dy, dx
        ax, xy, yx, ay = 0, ys, xs, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*ax + y*yx , y0 + x*xy + y*ay

        if D >= 0:
            y += 1
            D -= 2 * dx

    D += 2 * dy

