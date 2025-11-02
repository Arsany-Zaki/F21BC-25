from dataclasses import dataclass
from nn.nn import NNConfig
from pso.pso import PSOConfig
from exp_analysis import ExpAnalysis

@dataclass
class ExpConfig:
    nn_config: NNConfig
    pso_config: PSOConfig
    exp_analysis: ExpAnalysis