import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import numpy as np
import pytest
from src.pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO

# Dummy NNParams class to simulate the required attribute
class DummyNNParams:
	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes

def call_pso_vector_to_nn_weights(nn_params, flat_vector):
	trainer = NNTrainerUsingPSO(training_data={}, exp_params={'pso_params': None, 'nn_params': nn_params})
	return trainer._pso_vector_to_nn_weights(np.array(flat_vector))

test_cases = [
	{
		'layer_sizes': [2, 3],
		'input': [1, 2, 3, 4, 5, 6, 7, 8, 9],
		'expected_weights': [
			[
				[1, 4],
				[2, 5],
				[3, 6],
			]
		],
		'expected_biases': [
			[7, 8, 9]
		]
	},
	{
		'layer_sizes': [3, 2],
		'input': [1, 2, 3, 4, 5, 6, 7, 8],
		'expected_weights': [
			[
				[1, 3, 5],
				[2, 4, 6],
			]
		],
		'expected_biases': [
			[7, 8]
		]
	},
	# 3-layer network: [2, 2, 2]
	{
		'layer_sizes': [2, 2, 2],
		# Layer 1: 2 inputs, 2 neurons: 2*2=4 weights, 2 biases
		# Layer 2: 2 inputs, 2 neurons: 2*2=4 weights, 2 biases
		# Total: 4+2+4+2=12
		'input': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
		# Layer 1: weights [[1,3],[2,4]], biases [5,6]
		# Layer 2: weights [[7,9],[8,10]], biases [11,12]
		'expected_weights': [
			[
				[1, 3],
				[2, 4],
			],
			[
				[7, 9],
				[8, 10],
			]
		],
		'expected_biases': [
			[5, 6],
			[11, 12]
		]
	},
	# Add more cases as needed
]

@pytest.mark.parametrize("case", test_cases)
def test_pso_vector_to_nn_weights(case):
	nn_params = DummyNNParams(case['layer_sizes'])
	weights, biases = call_pso_vector_to_nn_weights(nn_params, case['input'])
	assert weights == case['expected_weights']
	assert biases == case['expected_biases']
