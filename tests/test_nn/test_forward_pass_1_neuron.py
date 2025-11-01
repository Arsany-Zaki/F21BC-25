from nn.nn import NeuralNetwork, NNConfig
from settings.enumerations import ActivationFunction, CostFunction


# Configuration for a neural network with no hidden layers (input layer directly connected to output layer)
config = NNConfig(
    input_dim = 3,
    layers_sizes = [1],
    activation_functions = [ActivationFunction.LINEAR], 
    cost_function = CostFunction.MEAN_SQUARED_ERROR
)

# Create the neural network
nn = NeuralNetwork(config)

# Define weights and biases for the output layer
weights = [
    [0.1, 0.2, 0.3],  # Weights for output neuron 1
]
biases = [0.5]  

input_points = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 2.0]
]

targets = [
    1.0,
    1.6,
    0.0
] 

expected_cost = 0
for features, target in zip(input_points, targets):
    prediction = sum(w * f for w, f in zip(weights[0], features)) + biases[0]
    expected_cost += (prediction - target) ** 2
expected_cost /= len(targets)
cost = nn.forward_pass(
    weights = [weights], 
    biases = [biases], 
    training_points = input_points,
    training_points_targets = targets
)   

print(f"Calculated cost : {cost}")
print(f"Correct cost    : {expected_cost}")

assert abs(cost - expected_cost) < 1e-6, f"Expected cost {expected_cost}, got {cost}"


