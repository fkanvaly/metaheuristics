# encoding: utf-8
from tqdm import tqdm
import yaml

from sho import make
from snp import get_args, get_solver

########################################################################
# Interface for multiple runs
########################################################################

if __name__ == "__main__":
    the = get_args()

    # get config
    solver_name = the.solver.split('_')[1]
    config = yaml.safe_load(open(f'config/{solver_name}/{the.param}.yml'))
    config["domain_width"] = the.domain_width
    config["nb_sensors"] = the.nb_sensors
    config["sensor_range"] = the.sensor_range

    for i in tqdm(range(the.nb_run)):
        writter = make.Writter(f"saves/{the.solver}.json", config=config, run=i, create=(i == 0 and the.save))
        solver, converter = get_solver(the, config, writter)
        val, sol = solver.run()

