from dataclasses import dataclass, field
from data_prep.constants import *

@dataclass
class DataPrepConfig:
    norm_method: NormMethod = NormMethod.ZSCORE
    norm_factors: list[float] = field(default_factory=lambda: NORM_DEFAULT_FACTORS[NormMethod.ZSCORE])
    split_test_size: float = 0.3    
    random_seed: int = 42           