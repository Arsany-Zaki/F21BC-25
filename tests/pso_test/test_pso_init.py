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

def make_params():
    return PSOParams(
        swarm_size=3,
        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        dims=2,
        bounds=[(-1, 1), (-1, 1)],
        max_iter=5,
        target_fitness=None,
        vel_limit=0.5,
        boundary_handling=BoundaryHandling.CLIP,
        informant_selection=InformantSelection.STATIC_RANDOM,
        informant_count=2
    )

def test_pso_initialization():
    pso = PSO(make_params())
    assert pso.positions.shape == (3, 2)
    assert pso.velocities.shape == (3, 2)
    assert pso.pbest_pos.shape == (3, 2)
    assert pso.gbest_pos is None
