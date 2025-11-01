import numpy as np
from typing import Any, Dict
from nn.nn import NeuralNetwork
from pso.pso import PSO, PSOConfig
from settings.activation_boundaries_settings import activation_boundaries
from settings.constants import *

class Analytics:
	fitness_calls_count: int = 0
	last_x_fitness_calls_sum: float = 0

class NNTrainerUsingPSO:
	def __init__(self, training_data, pso_config, nn_config):
		"""
		training_data: dict with keys 'inputs' and 'targets'
		pso_config: PSOConfig instance
		nn_config: NNConfig instance
		"""
		self.training_data = training_data
		self.pso_config = pso_config
		self.nn_config = nn_config
		self.nn = None
		self.pso = None
		self.analytics = Analytics()
	def train_nn_using_pso(self):
		# Create neural network
		self.nn = NeuralNetwork(
			config=self.nn_config
		)
		# Set PSO boundaries based on NN topology and activations
		boundaries = self._calculate_pso_feature_boundaries()
		self.pso_config.boundary_min = [b[0] for b in boundaries]
		self.pso_config.boundary_max = [b[1] for b in boundaries]
		# Create PSO
		self.pso = PSO(self.pso_config)
		# Run PSO optimize
		best_weights, best_fitness = self.pso.optimize(self._assess_fitness)
		print(f'Custom PSO completed. Best fitness: {best_fitness}')
		return best_weights, best_fitness

	def train_nn_using_pyswarm_pso(self):
		# Import pyswarm locally to avoid dependency if not used
		from pyswarm import pso as pyswarm_pso
		# Create neural network
		self.nn = NeuralNetwork(config=self.nn_config)
		# Set PSO boundaries based on NN topology and activations
		boundaries = self._calculate_pso_feature_boundaries()
		lb = [b[0] for b in boundaries]
		ub = [b[1] for b in boundaries]
		self.pso_config.boundary_min = lb
		self.pso_config.boundary_max = ub
		# Use pyswarm's pso optimizer
		best_weights, best_fitness = pyswarm_pso(
			self._assess_fitness,
			lb,
			ub,
			phig=self.pso_config.c_global,
			phip=self.pso_config.c_personal,
			omega=self.pso_config.w_inertia,
			swarmsize=self.pso_config.swarm_size
		)
		return best_weights, best_fitness

	def train_nn_pyswarm_pso_default(self):
		# Import pyswarm locally to avoid dependency if not used
		from pyswarm import pso as pyswarm_pso
		# Create neural network
		self.nn = NeuralNetwork(config=self.nn_config)
		# Set PSO boundaries based on NN topology and activations
		boundaries = self._calculate_pso_feature_boundaries()
		lower_bounds = [b[0] for b in boundaries]
		upper_bounds = [b[1] for b in boundaries]
		self.pso_config.boundary_min = lower_bounds
		self.pso_config.boundary_max = upper_bounds
		# Use pyswarm's pso optimizer
		best_weights, best_fitness = pyswarm_pso(self._assess_fitness,lower_bounds,upper_bounds)
		return best_weights, best_fitness

	def _assess_fitness(self, flat_weights: np.ndarray) -> float:
		
		# Convert flat vector to NN weights and biases structures
		weights_struct, biases_struct = self._pso_vector_to_nn_weights(flat_weights)
		# Run forward pass and return cost
		cost = self.nn.forward_pass(
			weights=weights_struct,
			biases=biases_struct,
			training_points=self.training_data['inputs'],
			training_points_targets=self.training_data['targets']
		)
		
		self.analytics.fitness_calls_count += 1
		self.analytics.last_x_fitness_calls_sum += cost
		if(self.analytics.fitness_calls_count % 100 == 0):
			print(f'Count: {self.analytics.fitness_calls_count} -> Fitness average: {self.analytics.last_x_fitness_calls_sum/100}')
			self.analytics.last_x_fitness_calls_sum = 0
		
		return cost

	def _pso_vector_to_nn_weights(self, flat_vector: np.ndarray
		) -> tuple[list[list[list[float]]], list[list[float]]]:
		"""
		Convert a flat vector to separate NN weights and biases structures:
		Returns (weights, biases) where:
		- weights: [layer][neuron][input_weight]
		- biases: [layer][neuron]
		The flat vector is assumed to be ordered as:
		- For each layer:
			- For each input: all neurons' weights for that input (input-major order)
			- All biases for all neurons in the layer (neuron-major order)
		Example for a layer with 2 inputs and 3 neurons:
		flat_vector = [w_0_0, w_0_1, w_0_2, w_1_0, w_1_1, w_1_2, b_0, b_1, b_2]
		where w_i_j is the weight from input i to neuron j, and b_j is the bias for neuron j.
		"""
		# layers_sizes: [hidden1, hidden2, ..., output] (does NOT include input layer)
		layer_sizes = self.nn_config.layers_sizes
		input_dim = self.nn_config.input_dim
		full_layer_sizes = [input_dim] + layer_sizes
		weights_struct = []
		biases_struct = []
		idx = 0
		for l in range(1, len(full_layer_sizes)):
			n_inputs = full_layer_sizes[l-1]
			n_neurons = full_layer_sizes[l]
			# Input weights: for each input, all neurons (input-major order)
			input_weights = []  # shape: [n_inputs][n_neurons]
			for i in range(n_inputs):
				input_weights.append(flat_vector[idx:idx+n_neurons].tolist())
				idx += n_neurons
			# Biases: one per neuron (neuron-major order)
			biases = flat_vector[idx:idx+n_neurons].tolist()
			idx += n_neurons
			# Build neuron-wise weights: [w1, w2, ...] for each neuron (no bias)
			layer_weights = []
			layer_biases = []
			for n in range(n_neurons):
				neuron_weights = [input_weights[i][n] for i in range(n_inputs)]
				layer_weights.append(neuron_weights)
				layer_biases.append(biases[n])
			weights_struct.append(layer_weights)
			biases_struct.append(layer_biases)
		return weights_struct, biases_struct

	def _calculate_pso_feature_boundaries(self):
		"""
		Use the instance's NN topology and activation functions to return a list of (lower, upper) tuples for each PSO feature (weight or bias), using boundaries from activation_boundaries_config.py.
		"""
		layer_sizes = self.nn_config.layers_sizes
		activation_functions = self.nn_config.activation_functions
		input_dim = self.nn_config.input_dim
		full_layer_sizes = [input_dim] + layer_sizes
		boundaries = []
		for l in range(1, len(full_layer_sizes)):
			n_inputs = full_layer_sizes[l-1]
			n_neurons = full_layer_sizes[l]
			act_fn = activation_functions[l-1]
			# Use boundaries from config with enum as key
			act_bounds = activation_boundaries[act_fn]
			weight_bounds = act_bounds[Constants.WEIGHT]
			bias_bounds = act_bounds[Constants.BIAS]
			for _ in range(n_inputs * n_neurons):
				boundaries.append(weight_bounds)
			for _ in range(n_neurons):
				boundaries.append(bias_bounds)
		return boundaries