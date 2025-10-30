from enum import Enum

class ActivationFunction(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'

class BoundaryHandling(Enum):
    CLIP = "clip"
    REFLECT = "reflect"

class CostFunction(Enum):
    MEAN_SQUARED_ERROR = 'mse'
    MEAN_ABSOLUTE_ERROR = 'mae'

class InformantSelection(Enum):
    STATIC_RANDOM = "static_random"
    DYNAMIC_RANDOM = "dynamic_random"
    SPATIAL_PROXIMITY = "spatial_proximity"