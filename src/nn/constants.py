from enum import Enum

class CostFunc(Enum):
    MEAN_SQUARED_ERROR = 'mse'
    MEAN_ABSOLUTE_ERROR = 'mae'

class ActFunc(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'

activation_boundary_weight = {
    ActFunc.TANH: (-1.0, 1.0),
    ActFunc.RELU: (-2.0, 2.0),
    ActFunc.SIGMOID: (-1.0, 1.0),
    ActFunc.LINEAR: (-1, 1)
}

activation_boundary_bias = {
    ActFunc.TANH: (0, 0.1),
    ActFunc.RELU: (0.0, 0.1),
    ActFunc.SIGMOID: (0, 0.5),
    ActFunc.LINEAR: (-0.1, 0.1)
}