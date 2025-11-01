from nn.nn import NeuralNetwork, NNConfig
from settings.enumerations import ActivationFunction, CostFunction
import pytest

# nn topology with 2 neurons in hidden layer and 1 neuron in output layer
config = NNConfig(
    input_dim = 3,
    layers_sizes = [2, 1],
    activation_functions = [ActivationFunction.RELU, ActivationFunction.LINEAR], 
    cost_function = CostFunction.MEAN_SQUARED_ERROR
)

# Test case data
input_points = [
    [1.0, 0.6, 1.0], 
    [1.0, 1.0, 1.0], 
    [0.0, 0.0, 2.0]
]
targets = [1.0, 1.2, 0.0]

weights_layer0 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.5]]
biases_layer0 = [0.2, 0.1]

weights_layer1 = [[0.5, 0.8]]
biases_layer1 = [0.5]

def test_nn_forward_pass() -> None:
    nn = NeuralNetwork(config)
    weights = [weights_layer0, weights_layer1]
    biases = [biases_layer0, biases_layer1]
    nn_cost = nn.forward_pass(weights, biases, input_points, targets)
    correct_cost = cost()

    print(f"Calculated NN Cost  : {nn_cost}")
    print(f"Correct Cost        : {correct_cost}")

    assert nn_cost == pytest.approx(correct_cost, rel=1e-2)

def cost() -> float:
    nn_outputs = forward_pass_full_topology_all_points()
    cost = cost_calculation(nn_outputs)
    return cost

def cost_calculation(nn_outputs: list[float]) -> float:
    total_cost = 0.0
    if(config.cost_function == CostFunction.MEAN_SQUARED_ERROR):
        for output, target in zip(nn_outputs, targets):
            total_cost += (output - target) ** 2
        return total_cost / len(targets)
    else:
        raise ValueError("Unsupported cost function for this test.")

def forward_pass_full_topology_all_points() -> list[float]:
    return [forward_pass_topology(point) for point in input_points]

def forward_pass_topology(point) -> float:
    layer0_output = [
        forward_pass_neuron(point, weights_layer0[0], biases_layer0[0], ActivationFunction.RELU),
        forward_pass_neuron(point, weights_layer0[1], biases_layer0[1], ActivationFunction.RELU)
    ]
    layer1_output = forward_pass_neuron(layer0_output, weights_layer1[0], biases_layer1[0], ActivationFunction.LINEAR)
    return layer1_output

def forward_pass_neuron(p, w, b, act) -> float:
    s = sum(wi * pi for wi, pi in zip(w, p)) + b
    if act == ActivationFunction.RELU:
        return max(0, s)
    elif act == ActivationFunction.LINEAR:
        return s
    else:
        raise ValueError("Unsupported activation function for this test.")
    
if __name__ == "__main__":
    test_nn_forward_pass()
