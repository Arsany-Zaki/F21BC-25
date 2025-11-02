from dataclasses import dataclass
from configs.metadata import NormMethod

@dataclass
class DataPrepConfig:
    norm_method: NormMethod
    norm_factors: list[float]
    split_test_size: float = 0.3    # Default to 30% test size
    random_seed: int = 42           # Default random seed