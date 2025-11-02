import numpy as np
import pytest
from pso.pso_config import PSOConfig
from nn.nn import NNConfig
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from configs.metadata import ActFunc as act_func, CostFunc as cost_func, BoundHandling as bound_handling, InformantSelect as informant_selec

# test case 2 layers (not including input layer) 
# layer0 has 2 neurons, layer1 has 1 neurons
# input dim is 3

nn_config = NNConfig(
    input_dim = 3,
    layers_sizes = [2, 1],
    activation_functions = [act_func.RELU, act_func.LINEAR],
    cost_function = cost_func.MEAN_SQUARED_ERROR
)
pso_config = PSOConfig(
    max_iter = 10,
    swarm_size = 10,
    informant_count = 2,

    boundary_handling = bound_handling.REFLECT,
    informant_selection = informant_selec.STATIC_RANDOM,

    w_inertia = 0.73,
    c_personal = 1.0,
    c_social = 1.0,
    c_global = 1.0,
    jump_size = 1.0,

    dims = 3,                
    boundary_min = [],       
    boundary_max = [],       
    target_fitness = None,
)

weights: list[list[list[float]]] = [
    [  # Layer 0
        [0.1, 0.2, 0.4],    # Neuron 0 weights
        [0.3, 0.4, 0.7],    # Neuron 1 weights
    ],
    [  # Layer 1
        [0.5, 0.6]    # Neuron 0 weights
    ]
]

biases = [[0.3, 0.5], [0.2]]

pso_vector = [0.1, 0.3, 0.2, 0.4, 0.4, 0.7, 0.3, 0.5, 0.5, 0.6, 0.2]

def test_pso_vector_to_nn_weights():
    trainer = NNTrainerUsingPSO(
        training_data={},
        pso_config=pso_config,
        nn_config=nn_config
    )
    weights_out, biases_out = trainer._pso_vector_to_nn_weights(np.array(pso_vector))

    # Check weights
    for layer_idx in range(len(weights)):
        for neuron_idx in range(len(weights[layer_idx])):
            for weight_idx in range(len(weights[layer_idx][neuron_idx])):
                actual = weights_out[layer_idx][neuron_idx][weight_idx]
                expected = weights[layer_idx][neuron_idx][weight_idx]
                print(f"Weight [L{layer_idx}][N{neuron_idx}][W{weight_idx}]: actual={actual}, expected={expected}")
                assert actual == pytest.approx(expected, rel=1e-6)

    # Check biases
    for layer_idx in range(len(biases)):
        for neuron_idx in range(len(biases[layer_idx])):
            actual = biases_out[layer_idx][neuron_idx]
            expected = biases[layer_idx][neuron_idx]
            print(f"Bias [L{layer_idx}][N{neuron_idx}]: actual={actual}, expected={expected}")
            assert actual == pytest.approx(expected, rel=1e-6)

if __name__ == "__main__":
    test_pso_vector_to_nn_weights()

    