"""Wrappers that captures parameters of a function
and returns an operator with a given interface."""
import json


def func(cover, **kwargs):
    """Make an objective function from the given function.
    An objective function takes a solution and returns a scalar."""

    def f(sol):
        return cover(sol, **kwargs)

    return f


def init(init, **kwargs):
    """Make an initialization operator from the given function.
    An init. op. returns a solution."""

    def f():
        return init(**kwargs)

    return f


def neig(neighb, **kwargs):
    """Make an neighborhood operator from the given function.
    A neighb. op. takes a solution and returns another one."""

    def f(sol):
        return neighb(sol, **kwargs)

    return f


def cross(crossover, **kwargs):
    """Make an neighborhood operator from the given function.
    A neighb. op. takes a solution and returns another one."""

    def f(par1, par2):
        return crossover(par1, par2, **kwargs)

    return f


def iter(iters, **kwargs):
    """Make an iterations operator from the given function.
    A iter. op. takes a value and a solution and returns
    the current number of iterations."""

    def f(i, val, sol):
        return iters(i, val, sol, **kwargs)

    return f


def sol_check(checker, **kwargs):
    def f(sol):
        return checker(sol, **kwargs)

    return f


def correction(proj, **kwargs):
    def f(sol):
        return proj(sol, **kwargs)

    return f


class Writter:
    def __init__(self, filepath, config, run=0, create=False):
        self.filepath = filepath
        self.run = str(run)

        if create:
            with open(self.filepath, "w") as f:
                json.dump({"config": config, "runs": {}}, f)

    def add_stat(self, i, val):
        with open(self.filepath, 'r') as fp:
            data = json.load(fp)

        if self.run not in data["runs"].keys():
            data["runs"][self.run] = [[i, val]]
        else:
            data["runs"][self.run].append([i, val])

        with open(self.filepath, 'w') as fp:
            json.dump(data, fp)


class call_tracker:
    def __init__(self, func, pen, writter: Writter, save=False):
        self.f = func
        self.pen = pen
        self.call = 0
        self.writter = writter
        self.best_val = None
        self.best_sol = None
        self.save = save

    def __call__(self, sol):
        val = self.f(sol) - self.pen(sol)
        if self.best_val is None:
            self.best_val = val
            self.best_sol = sol
            if self.save:
                self.writter.add_stat(self.call, self.best_val)
        else:
            if self.best_val < val:
                self.best_val = val
                self.best_sol = sol
                if self.save:
                    self.writter.add_stat(self.call, self.best_val)

        self.call += 1
        return val

    def get_best(self):
        return self.best_val, self.best_sol

    def get_call(self):
        return self.call


import numpy as np


######## RECUIT SIMULÉ #######

#  initial temperature
def init_tempareture(init_temperature, **kwargs):
    def f(**kwargs_spec):
        return init_temperature(**{**kwargs, **kwargs_spec})

    return f


def temp_init_type1(init, func, neighb):
    sol = init()
    val = func(sol)
    mean_T = np.mean([np.abs(val - func(neighb(sol))) for i in range(100)])
    accept_rate = 0.8
    T_init = - mean_T / np.log(accept_rate)
    return T_init


def temp_init_default(init, func, neighb, T):
    return T


#  decrease temperature
def decrease_temp(alpha, **kwargs):
    def f(T):
        return T * alpha(**kwargs)

    return f


def alpha_static(factor):
    return factor


#  penanlize

def penealize(pen, **kwargs):
    def f(sol):
        return pen(sol, **kwargs)

    return f
