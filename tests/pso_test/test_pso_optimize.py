import sys
import os
# Add src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import numpy as np
from pso.pso import PSO, PSOParams
from config.boundary_handling_enum import BoundaryHandling
from config.informant_selection_enum import InformantSelection

def sphere(x):
    return np.sum(x**2)

def make_params():
    return PSOParams(
        swarm_size=5,
        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        dims=2,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        target_fitness=0.01,
        vel_limit=0.5,
        boundary_handling=BoundaryHandling.CLIP,
        informant_selection=InformantSelection.STATIC_RANDOM,
        informant_count=2
    )

def test_optimize_runs():
    pso = PSO(make_params())
    best_pos, best_fit = pso.optimize(sphere)
    assert isinstance(best_pos, np.ndarray)
    assert isinstance(best_fit, float)
