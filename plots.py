# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from os.path import isfile, join
import json


def prob(history, t0, x_star):
    count = 0
    N = len(history)
    for i in range(N):
        times = history[i][:, 0]
        vals = history[i][:, 1]
        if max(times) <= t0:
            idx = np.argmax(times)
        else:
            idx = max(np.where(times <= t0)[0])

        count += int(vals[idx] >= x_star)

    return count / N


def get_iter(history, t):
    """
    return : [t,  t,  t,  t,  t]
             [x1, x2, x3, x4, x5]
    """
    data = []
    for i in range(len(history)):
        times = history[i][:, 0]
        vals = history[i][:, 1]
        if max(times) <= t:
            idx = np.argmax(times)
        else:
            idx = max(np.where(times <= t)[0])
        data.append(vals[idx])

    return np.vstack(([int(t)] * len(history), data))


# Vectorize the function
vprob = np.vectorize(prob)


########################################################################
# Single Solver
########################################################################

def plot_heat_ert(hist, time_max=1000, f_ax=None):
    if f_ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        f, ax = f_ax

    history = {int(k): np.array(v) for k, v in hist["runs"].items()}
    domain_width = hist['config']['domain_width']

    # HEAT
    dx, dy = 10, 10

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(500, domain_width ** 2 + dy, dy),
                    slice(1, time_max + dx, dx)]

    z = vprob(history, x, y)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('YlGnBu')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    f.colorbar(im, ax=ax)
    ax.set_title('ecdf heatmap', fontsize=10)
    ax.set_xlabel('number of call of objective function', fontsize=10)
    ax.set_ylabel('objective function', fontsize=10)


def plot_step_ert(hist, target, time_max=1000, f_ax=None):
    if f_ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        f, ax = f_ax

    history = {int(k): np.array(v) for k, v in hist["runs"].items()}

    T = np.arange(0, time_max, time_max // 100)
    p = vprob(history, T, target)

    ax.step(T, p, where="post", label=str(target))
    ax.set_title(f"solver : {hist['solver']} | runs: {len(history)} | erc_ecdf for target: {target}",
                 fontsize=10)
    ax.set_xlabel('number of call of objective function', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend()


def plot_convergence(hist, time_max=1000, f_ax=None, is_comparison=False):
    if f_ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        f, ax = f_ax

    history = {int(k): np.array(v) for k, v in hist["runs"].items()}

    T = np.arange(0, time_max, time_max // 20)
    df = []
    for t in T:
        df.append(get_iter(history, t))

    df = np.hstack(df)
    # Set up the matplotlib figure
    ax.set_title('distribution of solution over time', fontsize=10)
    b = sns.boxplot(x=df[0], y=df[1], ax=ax)
    b.tick_params(labelsize=5)
    ax.set_ylabel('objective function value', fontsize=10)
    ax.set_xlabel('nb of call', fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if is_comparison:
        ax.set_ylim(0, hist['config']['domain_width'] ** 2)

    plt.tight_layout()


def plot_full(hist, target=770, time_max=1000):
    sns.set()
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 6))
    sns.despine(left=True)

    # HEAT
    plot_heat_ert(hist, time_max, (f, ax0))

    # STEP
    plot_step_ert(hist, target, time_max, (f, ax1))

    # Convergence
    plot_convergence(hist, time_max, (f, ax2))

    f.suptitle(f'full evaualtion of {hist["solver"]}')
    plt.tight_layout()
    plt.show()


def plot_single(name, hist, target=770, time_max=1000):
    sns.set()
    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.despine(left=True)

    # HEAT
    if name == "heat":
        plot_heat_ert(hist, time_max, (f, ax))

    # STEP
    elif name == "ert":
        plot_step_ert(hist, target, time_max, (f, ax))

    # Convergence
    elif name == "conv":
        plot_convergence(hist, time_max, (f, ax))

    plt.tight_layout()
    plt.show()


########################################################################
# Compare two Solver
########################################################################
def compare_heat_ert(hist1, hist2, time_max=1000, f_ax=None):
    if f_ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        f, ax = f_ax

    domain_width = hist1['config']['domain_width']
    history1 = {int(k): np.array(v) for k, v in hist1["runs"].items()}
    solver1 = hist1["solver"]
    history2 = {int(k): np.array(v) for k, v in hist2["runs"].items()}
    solver2 = hist2["solver"]

    dx, dy = 10, 10
    y, x = np.mgrid[slice(500, domain_width ** 2 + dy, dy),
                    slice(1, time_max + dx, dx)]
    # HEAT
    g1 = vprob(history1, x, y)
    g2 = vprob(history2, x, y)

    dominate_by_1 = (g1 > g2) * (-0.3)
    dominate_by_2 = (g1 < g2) * (0.5)
    equal_positive = ((g1 == g2) * (g1 > 0) * (g2 > 0)) * (-1)
    equal_negative = ((g1 == g2) * (g1 == 0) * (g2 == 0)) * 0
    z = dominate_by_1 + dominate_by_2 + equal_negative + equal_positive

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(-1, 1)

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('jet')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(x, y, z, cmap='jet', norm=norm)
    # f.colorbar(im, ax=ax)
    ax.set_title(f' -- domination heatmap --\n'
                 f'light_blue:{solver1}, orange:{solver2} \n'
                 f'dark blue: equal, green: zero', fontsize=15)
    ax.set_xlabel('number of call of objective function', fontsize=10)
    ax.set_ylabel('objective function', fontsize=10)


def ert_compare(hist, target, time_max=1000, f_ax=None):
    if f_ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
    else:
        f, ax = f_ax

    history = {int(k): np.array(v) for k, v in hist["runs"].items()}

    T = np.arange(0, time_max, time_max // 100)
    p = vprob(history, T, target)

    ax.step(T, p, where="post", label=hist['solver'])
    ax.set_title(f"runs: {len(history)} | erc_ecdf for target: {target}",
                 fontsize=10)
    ax.set_xlabel('number of call of objective function', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend()


def compare_plot(name, hist_list, target=770, time_max=1000):
    sns.set()
    f, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.despine(left=True)

    # is solver comparable check TODO
    # domain
    # n_sensor
    # sensor range
    # runs
    # calls

    # STEP
    if name == "ert":
        ert_compare(hist_list[0], target, time_max, (f, ax))
        ert_compare(hist_list[1], target, time_max, (f, ax))

    if name == "heat":
        compare_heat_ert(hist_list[0], hist_list[1], target, (f, ax))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", nargs="+", default=["num_recuit"])
    parser.add_argument("--folder", default="saves/mine/", type=str)
    parser.add_argument("--target", default=790, type=int)

    parser.add_argument("-t", "--type", metavar="NAME", choices=["single", "compare"], default="single")
    parser.add_argument("-r", "--rep", metavar="REP", choices=["ert", "heat", "conv", "full"], default="full")

    the = parser.parse_args()

    if the.type == "single":
        assert len(the.solvers) == 1, f"need necessarly one solver. got : {len(the.solvers)}"

        with open(join(the.folder, the.solvers[0] + '.json'), 'r') as f:
            history_list = json.load(f)

        history_list["solver"] = the.solvers[0]
        if the.rep == "full":
            plot_full(history_list, target=the.target, time_max=2000)
        else:
            plot_single(the.rep, history_list, target=the.target, time_max=2000)

    if the.type == "compare":
        assert len(the.solvers) == 2, f"need necessarly two " \
                                      f"solvers to compare. got : {len(the.solvers)}"
        paths = [join(the.folder, f + '.json') for f
                 in the.solvers if isfile(join(the.folder, f + '.json'))]

        history_list = {0: None, 1: None}

        for i, solver_name in enumerate(the.solvers):
            with open(join(the.folder, solver_name + '.json'), 'r') as f:
                history_list[i] = json.load(f)
                history_list[i]["solver"] = solver_name

        compare_plot(the.rep, history_list, target=the.target, time_max=2000)
