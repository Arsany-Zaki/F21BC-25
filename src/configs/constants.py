from enum import Enum

class ActFunc(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'

class BoundHandling(Enum):
    CLIP = "clip"
    REFLECT = "reflect"

class CostFunc(Enum):
    MEAN_SQUARED_ERROR = 'mse'
    MEAN_ABSOLUTE_ERROR = 'mae'

class InformantSelect(Enum):
    STATIC_RANDOM = "static_random"
    DYNAMIC_RANDOM = "dynamic_random"
    SPATIAL_PROXIMITY = "spatial_proximity"

class NormMethod(Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"

input_data_columns = {
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
