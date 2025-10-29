# activation_boundaries_config.py

from config.activation_functions_enum import ActivationFunction

activation_boundaries = {
    ActivationFunction.TANH: {"weight": (-1.0, 1.0), "bias": (-1.0, 1.0)},
    ActivationFunction.RELU: {"weight": (0.0, 1.0), "bias": (0.0, 1.0)},
    ActivationFunction.SIGMOID: {"weight": (-6.0, 6.0), "bias": (-6.0, 6.0)},
    ActivationFunction.LINEAR: {"weight": (-10.0, 10.0), "bias": (-10.0, 10.0)}
    # Add more as needed
}
