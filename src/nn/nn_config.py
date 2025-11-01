from dataclasses import dataclass
from typing import List
from settings.enumerations import ActivationFunction, CostFunction

@dataclass
class NNConfig:
    input_dim: int | None
    layers_sizes: List[int]
    activation_functions: List[ActivationFunction]
    cost_function: CostFunction
