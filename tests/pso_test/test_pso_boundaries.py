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

def make_params(boundary_handling):
    return PSOParams(
        swarm_size=2,
        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        dims=1,
        bounds=[(0, 1)],
        max_iter=2,
        target_fitness=None,
        vel_limit=0.5,
        boundary_handling=boundary_handling,
        informant_selection=InformantSelection.STATIC_RANDOM,
        informant_count=1
    )

def test_clip_boundary():
    pso = PSO(make_params(BoundaryHandling.CLIP))
    pso.positions[:] = -1  # force out of bounds
    pso._apply_boundary_strategy()
    assert np.all(pso.positions >= 0)
    pso.positions[:] = 2
    pso._apply_boundary_strategy()
    assert np.all(pso.positions <= 1)

def test_reflect_boundary():
    pso = PSO(make_params(BoundaryHandling.REFLECT))
    pso.positions[:] = -1
    pso.velocities[:] = 1
    pso._apply_boundary_strategy()
    assert np.all(pso.positions >= 0)
    pso.positions[:] = 2
    pso.velocities[:] = -1
    pso._apply_boundary_strategy()
    assert np.all(pso.positions <= 1)
