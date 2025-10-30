import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import numpy as np
from tabulate import tabulate
from pso.pso import PSO
from pso.pso_config import PSOConfig
from pso.pyswarm_pso import PySwarmPSO
from settings.enumerations import BoundaryHandling, InformantSelection
from pyswarm import pso as pyswarm_pso

def sphere(x):
	return np.sum(np.square(x))

def main():
	dims = 5
	config = PSOConfig(
		max_iter=100,
        swarm_size=10,
		
        w_inertia=0.5,
		c_personal=0.5,
		c_social=0.0,
		c_global=0.5,
		jump_size=1.0,
		
		dims=dims,
		boundary_min=[-5.0]*dims,
		boundary_max=[5.0]*dims,
		
		boundary_handling=BoundaryHandling.CLIP,
		informant_selection=InformantSelection.STATIC_RANDOM,
		informant_count=3,
		
		target_fitness=None,
	)

	# Custom PSO
	pso = PSO(config)
	_, best_fit_custom = pso.optimize(sphere)

	# PySwarm PSO
	_, fopt = pyswarm_pso(
        sphere,
        
        swarmsize=config.swarm_size,
        maxiter=config.max_iter,
		
        omega=config.w_inertia,
        phip=config.c_personal,
        phig=config.c_global,
		
        lb=config.boundary_min,
        ub=config.boundary_max,

        debug=False
	)

	# PySwarm PSO with default config
	_, fopt_default = pyswarm_pso(
        sphere,
        lb = config.boundary_min,
        ub = config.boundary_max,
        swarmsize = config.swarm_size,
        maxiter = config.max_iter,
		
        debug=False
    )

	# Actual optimal for sphere is 0
	table = [
		["Custom PSO", best_fit_custom],
		["PySwarm PSO", fopt],
		["PySwarm PSO Default", fopt_default],
		["Optimal (Sphere)", 0.0]
	]
	print(tabulate(table, headers=["Algorithm", "Best Fitness"], floatfmt=".6f"))

if __name__ == "__main__":
	main()
