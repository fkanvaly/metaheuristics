import json
import os
from os.path import isfile, join

import numpy as np
from texttable import Texttable

from plots import prob


def area_under_curve(history, start, target, tmax=1000):
    hist = {int(k): np.array(v) for k, v in history.items()}
    vprob = np.vectorize(prob)
    T = np.arange(0, tmax, tmax // 10)
    the_range = range(start, target, (target - start) // 10)
    score = 0
    for x in the_range:
        p = vprob(hist, T, x)
        score += np.sum(p)
    return score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", nargs="+", default=[])
    parser.add_argument("--folder",  default="saves/", type=str)

    the = parser.parse_args()

    t = Texttable()
    solvers_result = []
    paths = [join(the.folder, f + '.json') for f in the.solvers if isfile(join(the.folder, f + '.json'))]
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as f:
            history = json.load(f)
        target = history['config']['domain_width'] ** 2
        score = area_under_curve(history["runs"], start=0, target=target)
        solvers_result.append([name, score])

    sorted_idx = np.array(solvers_result)[:, 1].argsort()[::-1]

    t.add_rows([['Solver', 'Score']] + np.array(solvers_result)[sorted_idx].tolist())
    print(t.draw())

    best = solvers_result[sorted_idx[0]]
    print('best solver is %s :%.2f'%(best[0], best[1]))
