from dataclasses import dataclass
from typing import Optional, Tuple, List
from settings.enumerations import *


@dataclass
class PSOConfig:
    max_iter: int

    swarm_size: int
    w_inertia: float    
    c_personal: float   
    c_social: float     # informant influence
    c_global: float     # global best influence
    jump_size: float
    informant_selection: InformantSelection
    informant_count: int
    boundary_handling: BoundaryHandling

    dims: int
    boundary_min: List[float]
    boundary_max: List[float]
    target_fitness: Optional[float]