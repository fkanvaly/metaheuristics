########################################################################
# Algorithms
########################################################################
import numpy as np
from sho import make, num, bit

class Recuit:
    def __init__(self, config, func, init, neighb, again, sol_check, correction=None, penalize=None):
        self.func = func
        self.init = init
        self.neighb = neighb
        self.again = again
        self.sol_check = sol_check
        self.penalize = penalize if penalize is not None else lambda x: 0
        self.correction = correction if correction is not None else lambda x: x

        # temperature initializer
        init_t = config['init_temp']
        foo = getattr(make, f"temp_init_{init_t['type']}")
        self.init_temp = make.init_tempareture(foo, **init_t["param"])

        # decrease temperature
        decrease_t = config['decrease']
        foo = getattr(make, f"alpha_{decrease_t['type']}")
        self.decrease_temp = make.decrease_temp(foo, **decrease_t["param"])

    def run(self):
        # initial temperature
        T = self.init_temp(init=self.init, func=self.func, neighb=self.neighb)
        i = 1
        val, sol = self.func.get_best()
        sol_iter = self.init() if sol is None else sol
        val_iter = self.func(sol_iter) if val is None else val
        while self.again(self.func.get_call(), val_iter, sol_iter):
            sol = self.correction(self.neighb(sol_iter))
            val = self.func(sol)

            # acceptance good solution and bad one with respect to T
            if val > val_iter or (T > 0 and np.random.rand() < np.exp((val - val_iter) / T)):
                val_iter = val
                sol_iter = sol

            T = self.decrease_temp(T)
            i += 1
        best_val, best_sol = self.func.get_best()

        #assert self.sol_check(best_sol), f"solution contains sensor out of range.\n sol={best_sol}"

        return best_val, best_sol


class Genetic:
    def __init__(self, config, func, init, neighb, crossover, again, sol_check, correction=None, penalize=None):
        self.func = func
        self.init = init
        self.neighb = neighb
        self.again = again
        self.crossover = crossover
        self.sol_check = sol_check
        self.penalize = penalize if penalize is not None else lambda x: 0
        self.correction = correction if correction is not None else lambda x: x

        # temperature initializer
        self.pop_size = config['pop_size']
        self.mutation = config['mutation']
        self.elitism = config['elitism']

    def run(self):
        # Parameter
        elitism = self.elitism
        pop_size = self.pop_size
        mutation = self.mutation  # mutation rate

        # Initial population
        pop = [self.init() for i in range(pop_size)]

        i = 0
        best_iter_sol = pop[0]
        best_iter_val = self.func(best_iter_sol)

        while self.again(self.func.get_call(), best_iter_val, best_iter_sol):
            # Fitness function
            fitness = np.array([self.func(elt) - self.penalize(elt) for elt in pop])

            idx = fitness.argmax()
            best_iter_sol = pop[idx]
            best_iter_val = fitness[idx]

            # Selection
            order = fitness.argsort()
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(fitness))
            ranks = ranks + 1
            prob = ranks / np.sum(ranks)

            # Crossover
            par1 = [np.random.choice(range(pop_size), p=prob.tolist()) for i in range(pop_size)]
            par2 = [np.random.choice(range(pop_size), p=prob.tolist()) for i in range(pop_size)]
            offstring = [self.crossover(pop[u], pop[v]) for (u, v) in zip(par1, par2)]

            # Mutation
            for k in range(pop_size):
                if np.random.rand() < mutation:
                    offstring[k] = np.array(self.neighb(offstring[k]))

            # elitism
            pop_elit = [pop[i] for i in order[::-1][:elitism]]
            pop = pop_elit + offstring[:pop_size - elitism]

            # correction
            pop = [self.correction(elt) for elt in pop]
            i += 1

        best_val, best_sol = self.func.get_best()
        #assert self.sol_check(best_sol), f"solution contains sensor out of range.\n sol={best_sol}"

        return best_val, best_sol
