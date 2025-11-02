from dataclasses import dataclass
from typing import List
from configs.metadata import ActFunc, CostFunc

@dataclass
class NNConfig:
    input_dim: int | None
    layers_sizes: List[int]
    activation_functions: List[ActFunc]
    cost_function: CostFunc
