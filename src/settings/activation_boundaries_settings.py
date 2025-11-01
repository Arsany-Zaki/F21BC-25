from .enumerations import *
from .constants import Constants as cons

activation_boundaries = {
	ActivationFunction.TANH: {
		cons.WEIGHT: (-1.0, 1.0),
		cons.BIAS: (0, 0.1)
	},
	ActivationFunction.RELU: {
		cons.WEIGHT: (-2.0, 2.0),
		cons.BIAS: (0.0, 0.1)
	},
	ActivationFunction.SIGMOID: {
		cons.WEIGHT: (-1.0, 1.0),
		cons.BIAS: (0, 0.5)
	},
	ActivationFunction.LINEAR: {
		cons.WEIGHT: (-1, 1),
		cons.BIAS: (-0.1, 0.1)
	},
	# Add more activation functions and their boundaries as needed
}
