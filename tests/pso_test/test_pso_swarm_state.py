import sys
import os
# Add src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from pso.pso import PSO, PSOParams
from config.boundary_handling_enum import BoundaryHandling
from config.informant_selection_enum import InformantSelection
import numpy as np

def make_params():
    return PSOParams(
        swarm_size=2,
        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        dims=1,
        bounds=[(-1, 1)],
        max_iter=2,
        target_fitness=None,
        vel_limit=0.5,
        boundary_handling=BoundaryHandling.CLIP,
        informant_selection=InformantSelection.STATIC_RANDOM,
        informant_count=1
    )

def test_get_swarm_state():
    pso = PSO(make_params())
    state = pso.get_swarm_state()
    assert 'positions' in state
    assert 'velocities' in state
    assert 'fitness' in state
    assert 'pbest_pos' in state
    assert 'pbest_fit' in state
    assert 'sbest_pos' in state
    assert 'sbest_fit' in state
    assert 'gbest_pos' in state
    assert 'gbest_fit' in state
