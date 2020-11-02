import math
import numpy as np
import copy

from . import x, y, pb


########################################################################
# Objective functions
########################################################################

def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given array of bits."""
    assert (0 < sensor_range <= math.sqrt(2))
    assert (0 < domain_width)
    assert (dim > 0)
    assert (len(sol) >= dim)
    domain = np.zeros((domain_width, domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range * domain_width)
    s = np.sum(cov)
    assert (s >= len(sensors))
    return s


def to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    assert (len(sol) > 0)
    sensors = []
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                sensors.append((j, i))
    return sensors


########################################################################
# Initialization
########################################################################

def rand(domain_width, nb_sensors):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros((domain_width, domain_width))
    for x, y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


def cover_whole(domain_width, sensor_range, n_sensors):
    sensor_range_domain = domain_width*sensor_range
    # we make the assumption of 2D grid
    start = sensor_range_domain
    end = domain_width
    sensor_per_row = n_sensors**0.5
    x = np.arange(start, end, (end-start)/sensor_per_row)
    y = np.arange(start, end, (end-start)/sensor_per_row)

    xx, yy = np.meshgrid(x, y)
    xx = xx.astype(np.int)
    yy = yy.astype(np.int)
    pts = [[[xx[i][j], yy[i][j]] for i in range(xx.shape[0])] for j in range(yy.shape[1])]
    pts = np.array(pts).flatten()
    n_sensors_grid = pts.shape[0]//2
    pts = pts.reshape(n_sensors_grid, 2)

    domain = np.zeros((domain_width, domain_width))

    if n_sensors <= n_sensors_grid:
        for x, y in pts[:n_sensors]:
            domain[y][x] = 1
    else:
        for x, y in pts:
            domain[y][x] = 1

        for x, y in np.random.randint(0, domain_width, (n_sensors-n_sensors_grid, 2)):
            domain[y][x] = 1

    return domain


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random array by moving every ones to adjacent cells."""
    assert (0 < scale <= 1)
    # Copy, because Python pass by reference
    # and we may not want to alter the original solution.
    new = copy.copy(sol)
    for py in range(len(sol)):
        for px in range(len(sol[py])):
            # Indices order is (y,x) in order to match
            # coordinates of images (row,col).
            if sol[py][px] == 1:
                # Add a one somewhere around.
                w = (scale / 2) * domain_width
                ny = np.random.randint(py - w, py + w)
                nx = np.random.randint(px - w, px + w)
                ny = min(max(0, ny), domain_width - 1)
                nx = min(max(0, nx), domain_width - 1)
                if new[nx][ny] != 1:
                    new[py][px] = 0  # Remove original position.
                    new[ny][nx] = 1
                # else pass
    return new


def crossover(par1, par2):
    """ perform mirror crossover by perserving the number of sensor"""
    child = par1 * par2  # initialize with common position
    n_sensor = np.sum(par1) - np.sum(child) # sensor to get from parents

    # count number of sensor in each row
    axe = int(np.random.random() > 0.5)

    par1_axe_count = np.sum(par1, axis=0)
    par2_axe_count = np.sum(par2, axis=0)

    remain_sensor_from1 = n_sensor // 2
    remain_sensor_from2 = n_sensor - n_sensor // 2

    n_rows, n_cols = par1.shape
    # place sensor coming from parent 1
    for i in range(n_rows):
        if not remain_sensor_from1: break
        if par1_axe_count[i]:
            for j in range(n_cols):
                if not remain_sensor_from1: break
                u,v = (j, i) if axe==0 else (i, j)
                if par1[u, v] and not child[u, v]:
                    child[u, v] = 1
                    remain_sensor_from1 -= 1

    # place sensor coming from parent 2
    for i in range(n_rows):
        if not remain_sensor_from2: break
        if par2_axe_count[i]:
            for j in range(n_cols):
                if not remain_sensor_from2: break
                u,v = (j, i) if axe==0 else (i, j)
                if par2[u, v] and not child[u, v]:
                    child[u, v] = 1
                    remain_sensor_from2 -= 1
    return child


def valid_solution(sol, domain_width):
    # TODO :
    return True

def penalizer(sol, domain_width, scale):
    pen = 0
    #for coord in sol:
    #    pen += scale * (max(0, coord - domain_width) + max(0, -coord))
    return pen