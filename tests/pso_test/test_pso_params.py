import sys
import os
# Add src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
import pytest
from pso.pso import PSOParams
from config.boundary_handling_enum import BoundaryHandling
from config.informant_selection_enum import InformantSelection

def test_pso_params_creation():
    params = PSOParams(
        swarm_size=5,
        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        dims=2,
        bounds=[(-1, 1), (-1, 1)],
        max_iter=10,
        target_fitness=None,
        vel_limit=0.5,
        boundary_handling=BoundaryHandling.CLIP,
        informant_selection=InformantSelection.STATIC_RANDOM,
        informant_count=2
    )
    assert params.swarm_size == 5
    assert params.dims == 2
    assert params.bounds == [(-1, 1), (-1, 1)]
    assert params.boundary_handling == BoundaryHandling.CLIP
    assert params.informant_selection == InformantSelection.STATIC_RANDOM
