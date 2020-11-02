import sys
from collections import deque

########################################################################
# Stopping criterions
########################################################################

def max(i, val, sol, nb_it):
    """Stop after reaching nb_it iterations."""
    if i < nb_it:
        return True
    else:
        return False


def target(i, val, sol, target):
    """Stop after reaching target value."""
    if val < target:
        return True
    else:
        return False


class steady:
    """Stop if improvement is lesser than epsilon, in the last delta iterations."""

    def __init__(self, delta, epsilon = 0):
        self.epsilon = epsilon
        self.stuck = delta
        self.delta = delta
        self.delta_vals = deque()

    def __call__(self, i, val, sol):
        if i < self.delta: # Always wait the first delta iterations.
            self.delta_vals.append(val)
            return True

        else:
            #FILO stack.
            self.delta_vals.popleft()
            self.delta_vals.append(val)

            if val - self.delta_vals[0] <= self.epsilon:
                return False # Stop here.
            else:
                return True


# Stopping criterions that are actually just checkpoints.

def several(i, val, sol, agains):
    """several stopping criterions in one."""
    over = []
    for again in agains:
        over.append( again(i, val, sol) )
    return all(over)


def save(i, val, sol, filename="run.csv", fmt="{it} ; {val} ; {sol}\n"):
    """Save all iterations to a file."""
    # Append a line at the end of the file.
    with open(filename.format(it=i), 'a') as fd:
        fd.write( fmt.format(it=i, val=val, sol=sol) )
    return True # No incidence on termination.


def history(i, val, sol, history):
    history.append((val,sol))
    return True


def log(i, val, sol, fmt="{it} {val}\n"):
    """Print progress on stderr."""
    sys.stderr.write( fmt.format(it=i, val=val) )
    return True


def history_(n_call, val, sol, history):
    history.append([n_call, val])
    return True


