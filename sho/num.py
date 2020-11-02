import math
import numpy as np

from . import pb


########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.
def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    assert (len(sol) > 0)
    sensors = []
    for i in range(0, len(sol), 2):
        sensors.append((int(math.floor(sol[i])), int(math.floor(sol[i + 1]))))
    return sensors


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given vector."""
    assert (0 < sensor_range <= domain_width * math.sqrt(2))
    assert (0 < domain_width)
    assert (dim > 0)
    assert (len(sol) >= dim)
    domain = np.zeros((domain_width, domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range * domain_width)
    s = np.sum(cov)
    assert (s >= len(sensors))
    return s


########################################################################
# Initialization
########################################################################

def rand(dim, scale):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


def cover_whole(domain_width, sensor_range, dim):
    sensor_range_domain = domain_width * sensor_range
    # we make the assumption of 2D grid
    start = sensor_range_domain
    end = domain_width
    sensor_per_row = (dim // 2) ** 0.5
    x = np.arange(start, end, (end - start) / sensor_per_row)
    y = np.arange(start, end, (end - start) / sensor_per_row)

    xx, yy = np.meshgrid(x, y)
    pts = [[[xx[i][j], yy[i][j]] for i in range(xx.shape[0])] for j in range(yy.shape[1])]
    pts = np.array(pts).flatten()
    dim_sensor_grid = pts.shape[0]

    if dim <= dim_sensor_grid:
        return pts[:dim]
    else:
        return np.hstack((pts,
                          domain_width * np.random.random(dim - dim_sensor_grid)))


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale` in [0,1]
    as a fraction of the domain width around the given solution."""
    assert (0 < scale <= 1)
    side = domain_width * scale
    new = sol + (np.random.random(len(sol)) * side - side / 2)
    return new


def crossover(par1, par2):
    num_sensor = par1.shape[0] // 2
    axe = int(np.random.random() > 0.5)
    # sorted position by x coordinate
    A = sorted(par1.reshape(num_sensor, 2).tolist(), key=lambda p: p[axe])
    B = sorted(par2.reshape(num_sensor, 2).tolist(), key=lambda p: -p[axe])  # decrease order

    common = set(tuple(i) for i in A) & set(tuple(i) for i in B)
    child = list(list(elt) for elt in common)
    remaining = len(A) - len(child)

    n1 = remaining // 2
    n2 = remaining - n1
    for i in range(num_sensor):
        if n1 and A[i] not in child:
            child.append(A[i])
            n1 -= 1

    n2 += n1  # if A didn't have enough elt
    for i in range(num_sensor):
        if n2 and B[i] not in child:
            child.append(B[i])
            n2 -= 1

    assert len(child) == num_sensor
    return np.array([list(p) for p in child]).flatten()  # flatten


def valid_solution(sol, domain_width):
    for coord in sol:
        if not (0 <= coord < domain_width):
            return False
    return True


def penalizer(sol, domain_width, scale):
    pen = 0
    for coord in sol:
        pen += scale * (max(0, coord - domain_width) + max(0, -coord))
    return pen


def projection(sol, domain_width):
    proj_sol = np.zeros_like(sol)
    for i in range(sol.shape[0]):
        proj_sol[i] = min(max(0, sol[i]), domain_width-1)
    return proj_sol
