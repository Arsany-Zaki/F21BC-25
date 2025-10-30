from .enumerations import *
from .constants import *

activation_boundaries = {
	ActivationFunction.TANH: {
		WEIGHT: (-1.0, 1.0),
		BIAS: (0, 0.1)
	},
	ActivationFunction.RELU: {
		WEIGHT: (-2.0, 2.0),
		BIAS: (0.0, 0.1)
	},
	ActivationFunction.SIGMOID: {
		WEIGHT: (-1.0, 1.0),
		BIAS: (0, 0.5)
	},
	ActivationFunction.LINEAR: {
		WEIGHT: (-1, 1),
		BIAS: (-0.1, 0.1)
	},
	# Add more activation functions and their boundaries as needed
}
