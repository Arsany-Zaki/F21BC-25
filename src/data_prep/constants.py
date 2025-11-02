from enum import Enum

INPUT_DATA_COLUMNS = {
    1: 'cement',
    2: 'blast_furnace_slag',
    3: 'fly_ash',
    4: 'water',
    5: 'Superplasticizer',
    6: 'coarse_aggregate',
    7: 'fine_aggregate',
    8: 'age',
    9: 'concrete_compressive_strength',
}

class NormMethod(Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"

NORM_DEFAULT_FACTORS = {
    NormMethod.ZSCORE: [0.0, 1.0],
    NormMethod.MINMAX: [-0.1, 1.0]
}