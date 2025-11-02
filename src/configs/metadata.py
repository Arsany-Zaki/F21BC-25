from configs.constants import *

norm_default_factors = {
    NormMethod.ZSCORE: [0.0, 1.0],
    NormMethod.MINMAX: [-0.1, 1.0]
}
activation_boundary_weight = {
    ActFunc.TANH: (-1.0, 1.0),
    ActFunc.RELU: (-2.0, 2.0),
    ActFunc.SIGMOID: (-1.0, 1.0),
    ActFunc.LINEAR: (-1, 1),
    # Add more activation functions and their boundaries as needed
}

activation_boundary_bias = {
    ActFunc.TANH: (0, 0.1),
    ActFunc.RELU: (0.0, 0.1),
    ActFunc.SIGMOID: (0, 0.5),
    ActFunc.LINEAR: (-0.1, 0.1),
    # Add more activation functions and their boundaries as needed
}