import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_path = os.path.join(project_root, 'src')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import numpy as np
from pso.pso import PSO
from pso.pso_config import PSOConfig
from pso.pyswarm_pso import PySwarmPSO
from settings.enumerations import BoundaryHandling, InformantSelection
from experiments.pso_experiment_framework import ExperimentSet, Experiment, RunSet

# Optimization functions
def sphere(x):
    return np.sum(np.square(x))

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def make_config(dims, use_informants, for_pyswarm_default=False):
    # If use_informants: informant effect, else: global effect
    if for_pyswarm_default:
        # Use pyswarm defaults: omega=0.5, phip=0.5, phig=0.5, swarmsize=100, maxiter=100
        return PSOConfig(
            swarm_size=100,
            w_inertia=0.5,
            c_personal=0.5,
            c_social=0.5,
            c_global=0.5,
            jump_size=1.0,
            dims=dims,
            boundary_min=[-5.0]*dims,
            boundary_max=[5.0]*dims,
            max_iter=100,
            target_fitness=None,
            boundary_handling=BoundaryHandling.CLIP,
            informant_selection=InformantSelection.STATIC_RANDOM,
            informant_count=3
        )
    if use_informants:
        c_global = 0.0
        c_social = 1.5
        informant_selection = InformantSelection.STATIC_RANDOM
        informant_count = 3
    else:
        c_global = 1.5
        c_social = 0.0
        informant_selection = InformantSelection.STATIC_RANDOM
        informant_count = 3
    return PSOConfig(
        swarm_size=30,
        w_inertia=0.7,
        c_personal=1.5,
        c_social=c_social,
        c_global=c_global,
        jump_size=1.0,
        dims=dims,
        boundary_min=[-5.0]*dims,
        boundary_max=[5.0]*dims,
        max_iter=100,
        target_fitness=None,
        boundary_handling=BoundaryHandling.CLIP,
        informant_selection=informant_selection,
        informant_count=informant_count
    )

def main():
    n_runs = 10
    dims = 10
    functions = [
        ("Sphere", sphere),
        ("Rastrigin", rastrigin),
        ("Rosenbrock", rosenbrock)
    ]

    # Experiment Set 1: Global effect (no informants)
    exp_set_global = ExperimentSet("Global Effect (No Informants)")
    for fname, func in functions:
        config = make_config(dims, use_informants=False)
        exp = Experiment(fname, config, func)
        # Custom PSO
        exp.add_runset(RunSet("Custom PSO", PSO, config, func, n_runs=n_runs))
        # PySwarm PSO (with same params)
        exp.add_runset(RunSet("PySwarm PSO (same params)", PySwarmPSO, config, func, n_runs=n_runs))
        # PySwarm PSO (default params)
        exp.add_runset(RunSet("PySwarm PSO (default)", PySwarmPSO, make_config(dims, use_informants=False, for_pyswarm_default=True), func, n_runs=n_runs))
        exp_set_global.add_experiment(exp)

    # Experiment Set 2: Informant effect (no global)
    exp_set_informant = ExperimentSet("Informant Effect (No Global)")
    for fname, func in functions:
        config = make_config(dims, use_informants=True)
        exp = Experiment(fname, config, func)
        # Custom PSO
        exp.add_runset(RunSet("Custom PSO", PSO, config, func, n_runs=n_runs))
        # PySwarm PSO (with same params)
        exp.add_runset(RunSet("PySwarm PSO (same params)", PySwarmPSO, config, func, n_runs=n_runs))
        # PySwarm PSO (default params)
        exp.add_runset(RunSet("PySwarm PSO (default)", PySwarmPSO, make_config(dims, use_informants=True, for_pyswarm_default=True), func, n_runs=n_runs))
        exp_set_informant.add_experiment(exp)

    # Run and print results
    exp_set_global.run()
    exp_set_informant.run()

if __name__ == "__main__":
    main()
