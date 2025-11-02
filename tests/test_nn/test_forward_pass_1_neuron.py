from nn.nn import NeuralNetwork
from nn.nn_config import NNConfig
from configs.metadata import ActFunc, CostFunc

nn = NeuralNetwork(NNConfig(
    input_dim = 3,
    layers_sizes = [1],
    activation_functions = [ActFunc.LINEAR], 
    cost_function = CostFunc.MEAN_SQUARED_ERROR
))

weights = [[0.1, 0.2, 0.3]]
biases = [0.5]  
input_points = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 2.0]
]
targets = [1.0, 1.6, 0.0] 

neuron_output = [sum(w * f for w, f in zip(weights[0], features)) + biases[0] for features in input_points]
actual_cost = sum((prediction - target) ** 2 for prediction, target in zip(neuron_output, targets)) / len(targets)

expected_cost = nn.forward_pass(
    weights = [weights], 
    biases = [biases], 
    training_points = input_points,
    training_points_targets = targets
)   

print(f"Calculated cost : {expected_cost}")
print(f"Actual cost     : {actual_cost}")

assert abs(expected_cost - actual_cost) < 1e-6, f"Expected cost is {expected_cost} while actual cost is {actual_cost}"


