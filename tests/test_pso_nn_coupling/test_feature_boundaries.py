import numpy as np
import pytest
from pso.pso_config import PSOConfig
from nn.nn import NNConfig
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from settings.enumerations import ActivationFunction as act_func, CostFunction as cost_func, BoundaryHandling as bound_handling, InformantSelection as informant_selec
from settings.activation_boundaries_settings import activation_boundaries as act_bound
from settings.constants import *

# test case 2 layers (not including input layer) 
# layer0 has 2 neurons, layer1 has 1 neurons
# input dim is 3

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

nn_config = NNConfig(
    input_dim = 3,
    layers_sizes = [4, 2, 1],
    activation_functions = [act_func.SIGMOID, act_func.TANH, act_func.LINEAR],
    cost_function = cost_func.MEAN_SQUARED_ERROR
)

bound_layer0_w_max = np.full(12, act_bound[nn_config.activation_functions[0]][WEIGHT][1])
bound_layer0_b_max = np.full(4, act_bound[nn_config.activation_functions[0]][BIAS][1])
bound_layer1_w_max = np.full(8, act_bound[nn_config.activation_functions[1]][WEIGHT][1])
bound_layer1_b_max = np.full(2, act_bound[nn_config.activation_functions[1]][BIAS][1])
bound_layer2_w_max = np.full(2, act_bound[nn_config.activation_functions[2]][WEIGHT][1])
bound_layer2_b_max = np.full(1, act_bound[nn_config.activation_functions[2]][BIAS][1])

bound_max = np.concatenate([
    bound_layer0_w_max,
    bound_layer0_b_max,
    bound_layer1_w_max,
    bound_layer1_b_max,
    bound_layer2_w_max,
    bound_layer2_b_max
])

bound_layer0_w_min = np.full(12, act_bound[nn_config.activation_functions[0]][WEIGHT][0])
bound_layer0_b_min = np.full(4, act_bound[nn_config.activation_functions[0]][BIAS][0])
bound_layer1_w_min = np.full(8, act_bound[nn_config.activation_functions[1]][WEIGHT][0])
bound_layer1_b_min = np.full(2, act_bound[nn_config.activation_functions[1]][BIAS][0])
bound_layer2_w_min = np.full(2, act_bound[nn_config.activation_functions[2]][WEIGHT][0])
bound_layer2_b_min = np.full(1, act_bound[nn_config.activation_functions[2]][BIAS][0])

bound_min = np.concatenate([
    bound_layer0_w_min,
    bound_layer0_b_min,
    bound_layer1_w_min,
    bound_layer1_b_min,
    bound_layer2_w_min,
    bound_layer2_b_min
])

def test_feature_boundaries():
    trainer = NNTrainerUsingPSO(
        training_data = {},
        pso_config = pso_config,
        nn_config = nn_config
    )
    boundaries = trainer._calculate_pso_feature_boundaries()

    all_min = np.array([b[0] for b in boundaries])
    all_max = np.array([b[1] for b in boundaries])

    print('')
    print(f"Expected min boundaries : {bound_min}")
    print(f"Calculated min boundaries: {all_min}")
    print(f"Expected max boundaries : {bound_max}")
    print(f"Calculated max boundaries: {all_max}")

    assert np.array_equal(all_min, bound_min), "Minimum boundaries do not match expected values."
    assert np.array_equal(all_max, bound_max), "Maximum boundaries do not match expected values."

if __name__ == "__main__":
    test_feature_boundaries()