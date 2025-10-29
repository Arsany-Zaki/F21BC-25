# Imports
import numpy as np
from typing import Any, Dict
from nn.generic_nn import NeuralNetwork, ActivationFunction, CostFunction
from pso.pso import PSO, PSOParams
from config.activation_boundaries_config import activation_boundaries

# Main class
class NNTrainerUsingPSO:
	def __init__(self, training_data, exp_params: Dict[str, Any]):
		"""
		training_data: dict with keys 'inputs' and 'targets'
		exp_params: dict with keys 'pso_params' (PSOParams) and 'nn_params' (dict)
		"""
		self.training_data = training_data
		self.pso_params = exp_params['pso_params']
		self.nn_params = exp_params['nn_params']
		self.nn = None
		self.pso = None

	def train_nn(self):
		# Create neural network
		self.nn = NeuralNetwork(
			config=self.nn_params
		)
		# Set PSO boundaries based on NN topology and activations
		boundaries = self._calculate_pso_feature_boundaries()
		self.pso_params.bounds = boundaries
		# Create PSO
		self.pso = PSO(self.pso_params)
		# Run PSO optimize
		best_weights, best_fitness = self.pso.optimize(self.assess_fitness)
		return best_weights, best_fitness

	def assess_fitness(self, flat_weights: np.ndarray) -> float:
		# Convert flat vector to NN weights structure
		weights_struct = self._pso_vector_to_nn_weights(flat_weights)
		# Run forward pass and return cost
		cost = self.nn.forward_pass(
			weights=weights_struct,
			input_vectors=self.training_data['inputs'],
			target_outputs=self.training_data['targets']
		)
		print(f"Fitness (cost) evaluated: {cost}")
		return cost

	def _pso_vector_to_nn_weights(self, flat_vector: np.ndarray):
		"""
		Convert a flat vector to the NN weights structure expected by NeuralNetwork.forward_pass:
		[layer][neuron][weight] where first weight is bias, rest are input weights.
		The flat vector is assumed to be ordered as:
		- For each layer:
			- For each input: all neurons' weights for that input (input-major order)
			- All biases for all neurons in the layer (neuron-major order)
		Example for a layer with 2 inputs and 3 neurons:
		flat_vector = [w_0_0, w_0_1, w_0_2, w_1_0, w_1_1, w_1_2, b_0, b_1, b_2]
		where w_i_j is the weight from input i to neuron j, and b_j is the bias for neuron j.
		"""
		layer_sizes = self.nn_params.layer_sizes
		weights_struct = []
		idx = 0
		for l in range(1, len(layer_sizes)):
			n_inputs = layer_sizes[l-1]
			n_neurons = layer_sizes[l]
			# Input weights: for each input, all neurons (input-major order)
			input_weights = []  # shape: [n_inputs][n_neurons]
			for i in range(n_inputs):
				input_weights.append(flat_vector[idx:idx+n_neurons].tolist())
				idx += n_neurons
			# Biases: one per neuron (neuron-major order)
			biases = flat_vector[idx:idx+n_neurons].tolist()
			idx += n_neurons
			# Build neuron-wise weights: [bias, w1, w2, ...] for each neuron
			layer_weights = []
			for n in range(n_neurons):
				neuron_weights = [biases[n]] + [input_weights[i][n] for i in range(n_inputs)]
				layer_weights.append(neuron_weights)
			weights_struct.append(layer_weights)
		return weights_struct

	def _calculate_pso_feature_boundaries(self):
		"""
		Use the instance's NN topology and activation functions to return a list of (lower, upper) tuples for each PSO feature (weight or bias), using boundaries from activation_boundaries_config.py.
		"""
		layer_sizes = self.nn_params.layer_sizes
		activation_functions = self.nn_params.activation_functions
		boundaries = []
		for l in range(1, len(layer_sizes)):
			n_inputs = layer_sizes[l-1]
			n_neurons = layer_sizes[l]
			act_fn = activation_functions[l-1]
			# Use boundaries from config with enum as key
			act_bounds = activation_boundaries[act_fn]
			weight_bounds = act_bounds['weight']
			bias_bounds = act_bounds['bias']
			for _ in range(n_inputs * n_neurons):
				boundaries.append(weight_bounds)
			for _ in range(n_neurons):
				boundaries.append(bias_bounds)
		return boundaries