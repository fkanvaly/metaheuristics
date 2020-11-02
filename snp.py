# encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml

from sho import make, algo, iters, plot, num, bit, pb


########################################################################
# Interface
########################################################################

def get_args():
    import argparse

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
                     help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
                     help="Sensors' range (as a fraction of domain width, max is √2)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
                     help="Domain width (a number of cells). If you change this you will probably need to update "
                          "`--target` accordingly")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
                     help="Maximum number of iterations")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
                     help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_recuit", "bit_recuit", "num_genetic", "bit_genetic"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="num_recuit",
                     help="Solver to use, among: " + ", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30 * 30, type=float,
                     help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
                     help="Stop if no improvement after NB iterations")

    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
                     help="Stop if the improvement of the objective function value is lesser than DVAL")

    can.add_argument("-a", "--variation-scale", metavar="RATIO", default=0.3, type=float,
                     help="Scale of the variation operators (as a ration of the domain width)")

    can.add_argument("-p", "--param", metavar="CONFIG", type=str,
                     help="model parameter")

    can.add_argument("--nb-run", metavar="NRUN", default=3, type=int,
                     help="Number of runs")

    can.add_argument("--call", metavar="CALL", default=2000, type=int,
                     help="max number of call of func")

    can.add_argument("--save", dest='save', action='store_true')

    can.add_argument("--id", metavar="ID", default=0, type=int,
                     help="id of save")

    the = can.parse_args()

    # Minimum checks.
    assert (0 < the.nb_sensors)
    assert (0 < the.sensor_range <= math.sqrt(2))
    assert (0 < the.domain_width)
    assert (0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth=np.inf)

    return the


def get_solver(the, config, writter):
    again = make.iter(
        iters.several,
        agains=[
            make.iter(iters.max,
                      nb_it=the.call),
            # make.iter(iters.target,
            #          target=the.target),
            make.iter(iters.log,
                      fmt="\r{it} {val}")
        ]
    )

    # Dimension of the search space.
    d = 2
    solver = None
    converter = None
    if the.solver == "num_recuit":
        solver = algo.Recuit(config=config,
                             func=make.call_tracker(make.func(num.cover_sum,
                                                              domain_width=the.domain_width,
                                                              sensor_range=the.sensor_range,
                                                              dim=d * the.nb_sensors),
                                                    make.penealize(num.penalizer,
                                                                   domain_width=the.domain_width,
                                                                   scale=0),
                                                    writter,
                                                    save=the.save),
                             init=make.init(num.cover_whole,
                                            dim=d * the.nb_sensors,
                                            domain_width=the.domain_width,
                                            sensor_range=the.sensor_range),
                             neighb=make.neig(num.neighb_square,
                                              scale=config["variation_scale"],
                                              domain_width=the.domain_width),
                             again=again,
                             sol_check=make.sol_check(num.valid_solution, domain_width=the.domain_width),
                             correction=make.correction(num.projection, domain_width=the.domain_width)
                             )
        converter = num.to_sensors

    if the.solver == "bit_recuit":
        solver = algo.Recuit(config=config,
                             func=make.call_tracker(make.func(bit.cover_sum,
                                                              domain_width=the.domain_width,
                                                              sensor_range=the.sensor_range,
                                                              dim=d * the.nb_sensors),
                                                    make.penealize(bit.penalizer,
                                                                   domain_width=the.domain_width,
                                                                   scale=1000),
                                                    writter,
                                                    save=the.save),
                             init=make.init(bit.cover_whole,
                                            n_sensors=the.nb_sensors,
                                            domain_width=the.domain_width,
                                            sensor_range=the.sensor_range),
                             neighb=make.neig(bit.neighb_square,
                                              scale=config["variation_scale"],
                                              domain_width=the.domain_width),
                             again=again,
                             sol_check=make.sol_check(bit.valid_solution, domain_width=the.domain_width),
                             )
        converter = bit.to_sensors

    """############# ALGO GENETIC #############"""
    if the.solver == "num_genetic":
        solver = algo.Genetic(config=config,
                              func=make.call_tracker(make.func(num.cover_sum,
                                                               domain_width=the.domain_width,
                                                               sensor_range=the.sensor_range,
                                                               dim=d * the.nb_sensors),
                                                     make.penealize(num.penalizer,
                                                                    domain_width=the.domain_width,
                                                                    scale=0),
                                                     writter,
                                                     save=the.save),
                              init=make.init(num.rand,
                                             dim=d * the.nb_sensors,
                                             scale=the.domain_width),
                              neighb=make.neig(num.neighb_square,
                                               scale=config["variation_scale"],
                                               domain_width=the.domain_width),
                              crossover=make.cross(num.crossover),
                              sol_check=make.sol_check(num.valid_solution, domain_width=the.domain_width),
                              again=again,
                              correction=make.correction(num.projection, domain_width=the.domain_width)
                              )
        converter = num.to_sensors

    elif the.solver == "bit_genetic":
        solver = algo.Genetic(config=config,
                              func=make.call_tracker(make.func(bit.cover_sum,
                                                               domain_width=the.domain_width,
                                                               sensor_range=the.sensor_range,
                                                               dim=d * the.nb_sensors),
                                                     make.penealize(bit.penalizer,
                                                                    domain_width=the.domain_width,
                                                                    scale=0),
                                                     writter,
                                                     save=the.save),
                              init=make.init(bit.rand,
                                             domain_width=the.domain_width,
                                             nb_sensors=the.nb_sensors),
                              neighb=make.neig(bit.neighb_square,
                                               scale=config["variation_scale"],
                                               domain_width=the.domain_width),
                              crossover=make.cross(bit.crossover),
                              sol_check=make.sol_check(bit.valid_solution, domain_width=the.domain_width),
                              again=again
                              )

        converter = bit.to_sensors

    return solver, converter


def plot_map(sensors, domain_width, sensor_range):
    """plot the grid and place sensor

    Args:
        sensors (ndarray): tuple of sensor place
        domain_width (int)
        sensor_range (float)
    """
    shape = (domain_width, domain_width)

    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
                         sensor_range * domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    the = get_args()

    # get config
    solver_name = the.solver.split('_')[1]
    config = yaml.safe_load(open(f'config/{solver_name}/{the.param}.yml'))
    config["domain_width"] = the.domain_width
    config["nb_sensors"] = the.nb_sensors
    config["sensor_range"] = the.sensor_range

    writter = None
    if the.save :
        writter = make.Writter(f"saves/{the.solver}.json", config=config, run=0, create= the.save)
    
    solver, converter = get_solver(the, config, writter=writter)
    val, sol = solver.run()
    sensors = converter(sol)

    # Fancy output.
    print("\n{} : {}".format(val, sensors))
    plot_map(sensors, the.domain_width, the.sensor_range)
