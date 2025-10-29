"""
Test module for Neural Network Forward Pass
Demonstrates initialization and forward pass with weights and inputs
"""

import sys
import os

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nn.neural_network import NeuralNetwork, ActivationFunction, CostFunction


def test_simple_hidden_layer():
    """
    Test network with 8 inputs -> 5 hidden -> 1 output
    """
    print("Testing Simple Hidden Layer Network (8->5->1)")
    print("=" * 60)
    
    # Define network topology: 8 inputs -> 5 hidden -> 1 output
    layer_sizes = [8, 5, 1]
    activation_functions = [ActivationFunction.SIGMOID, ActivationFunction.LINEAR]  # Hidden: sigmoid, Output: linear
    error_function = CostFunction.MEAN_SQUARED_ERROR
    
    # Initialize the neural network
    nn = NeuralNetwork(layer_sizes, activation_functions, error_function)
    
    print(f"Network topology: {layer_sizes}")
    print(f"Number of layers (excluding input): {len(nn.layers)}")
    
    # Define weights structure: [layer][neuron][weight]
    # Each neuron's weights: [bias, input1_weight, input2_weight, ...]
    weights = [
        # Hidden layer weights (5 neurons, each with bias + 8 input weights)
        [
            [0.5, 0.2, 0.3, -0.1, 0.4, -0.2, 0.1, 0.3, -0.15],   # Neuron 1
            [-0.1, 0.4, -0.2, 0.3, -0.1, 0.5, -0.3, 0.2, 0.1],   # Neuron 2
            [0.3, -0.3, 0.5, 0.2, -0.4, 0.1, 0.6, -0.2, 0.3],    # Neuron 3
            [0.2, 0.1, -0.4, 0.3, 0.2, -0.1, 0.4, 0.5, -0.3],    # Neuron 4
            [-0.2, 0.3, 0.1, -0.5, 0.2, 0.4, -0.1, 0.3, 0.2]     # Neuron 5
        ],
        # Output layer weights (1 neuron with bias + 5 input weights from hidden layer)
        [
            [0.1, 0.6, -0.4, 0.2, 0.3, -0.5]  # Output neuron
        ]
    ]
    
    # Define input vectors (8 inputs each) and expected outputs
    input_vectors = [
        [1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6],   # Input vector 1
        [0.3, 0.8, 0.1, 0.9, 0.5, 0.2, 0.7, 0.4],   # Input vector 2
        [0.7, 0.2, 0.6, 0.1, 0.8, 0.3, 0.5, 0.9],   # Input vector 3
        [0.9, 0.4, 0.2, 0.6, 0.1, 0.8, 0.3, 0.5]    # Input vector 4
    ]
    
    target_outputs = [0.8, 0.4, 0.6, 0.9]  # Expected outputs for each input
    
    print(f"\nInput vectors: {input_vectors}")
    print(f"Target outputs: {target_outputs}")
    
    # Perform forward pass and get total cost
    total_cost = nn.forward_pass(weights, input_vectors, target_outputs)
    
    print(f"\nTotal cost (MSE): {total_cost:.6f}")
    
    # Test individual predictions
    print("\nIndividual predictions:")
    print("-" * 30)
    
    for i, (input_vec, target) in enumerate(zip(input_vectors, target_outputs)):
        # Forward pass for single input
        current_inputs = input_vec
        
        for layer_idx, layer in enumerate(nn.layers):
            layer_weights = weights[layer_idx]
            current_inputs = layer.forward(current_inputs, layer_weights)
        
        prediction = current_inputs[0]
        error = (prediction - target) ** 2
        
        print(f"Input: {input_vec}, Target: {target:.3f}, Prediction: {prediction:.6f}, Error: {error:.6f}")
    
    return total_cost


def test_multi_hidden_layers():
    """
    Test network with 8 inputs -> 10 hidden -> 6 hidden -> 3 hidden -> 1 output
    """
    print("\n\nTesting Multi Hidden Layers Network (8->10->6->3->1)")
    print("=" * 60)
    
    # Define network topology: 8 inputs -> 10 -> 6 -> 3 -> 1 output
    layer_sizes = [8, 10, 6, 3, 1]
    activation_functions = [ActivationFunction.RELU, ActivationFunction.SIGMOID, ActivationFunction.TANH, ActivationFunction.LINEAR]  # Multiple activation functions
    error_function = CostFunction.MEAN_ABSOLUTE_ERROR
    
    # Initialize the neural network
    nn = NeuralNetwork(layer_sizes, activation_functions, error_function)
    
    print(f"Network topology: {layer_sizes}")
    print(f"Number of layers (excluding input): {len(nn.layers)}")
    
    # Define weights for this topology
    weights = [
        # First hidden layer (10 neurons, each with bias + 8 input weights)
        [
            [0.1, 0.2, 0.3, 0.4, -0.1, 0.5, -0.2, 0.3, 0.1],      # Neuron 1
            [-0.1, 0.5, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5],    # Neuron 2
            [0.2, -0.3, 0.4, -0.1, 0.6, 0.2, -0.4, 0.1, -0.2],    # Neuron 3
            [0.3, 0.1, -0.4, 0.2, -0.3, 0.5, 0.1, 0.4, -0.1],     # Neuron 4
            [-0.2, 0.4, 0.1, -0.5, 0.2, -0.1, 0.6, -0.3, 0.2],    # Neuron 5
            [0.4, -0.2, 0.3, 0.1, -0.4, 0.3, 0.2, -0.5, 0.1],     # Neuron 6
            [0.1, 0.3, -0.1, 0.4, 0.2, -0.3, 0.5, 0.1, -0.4],     # Neuron 7
            [-0.3, 0.2, 0.5, -0.1, 0.3, 0.4, -0.2, 0.1, 0.3],     # Neuron 8
            [0.2, -0.4, 0.1, 0.3, -0.2, 0.1, 0.4, -0.3, 0.5],     # Neuron 9
            [0.5, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, 0.2, -0.4]      # Neuron 10
        ],
        # Second hidden layer (6 neurons, each with bias + 10 input weights)
        [
            [0.1, 0.3, -0.2, 0.4, -0.1, 0.5, 0.2, -0.3, 0.1, 0.4, -0.2],  # Neuron 1
            [-0.2, 0.1, 0.5, -0.3, 0.2, -0.1, 0.4, 0.3, -0.2, 0.1, 0.5],  # Neuron 2
            [0.3, -0.1, 0.2, 0.4, -0.3, 0.1, -0.2, 0.5, 0.3, -0.4, 0.1],  # Neuron 3
            [0.2, 0.4, -0.3, 0.1, 0.2, -0.5, 0.3, 0.1, -0.2, 0.4, -0.1],  # Neuron 4
            [-0.1, 0.2, 0.3, -0.4, 0.1, 0.5, -0.2, 0.3, 0.4, -0.1, 0.2],  # Neuron 5
            [0.4, -0.3, 0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3]   # Neuron 6
        ],
        # Third hidden layer (3 neurons, each with bias + 6 input weights)
        [
            [0.1, 0.4, -0.2, 0.3, 0.1, -0.4, 0.2],   # Neuron 1
            [-0.2, 0.1, 0.5, -0.3, 0.4, 0.2, -0.1],  # Neuron 2
            [0.3, -0.1, 0.2, 0.4, -0.3, 0.1, 0.5]    # Neuron 3
        ],
        # Output layer (1 neuron with bias + 3 input weights)
        [
            [0.05, 0.8, -0.6, 0.4]  # Output neuron
        ]
    ]
    
    # Define input vectors (8 inputs each) and expected outputs
    input_vectors = [
        [1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6],
        [0.2, 0.8, 0.6, 0.1, 0.9, 0.3, 0.5, 0.7],
        [0.7, 0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3]
    ]
    
    target_outputs = [0.7, 0.3, 0.8]
    
    print(f"\nInput vectors: {input_vectors}")
    print(f"Target outputs: {target_outputs}")
    
    # Perform forward pass and get total cost
    total_cost = nn.forward_pass(weights, input_vectors, target_outputs)
    
    print(f"\nTotal cost (MAE): {total_cost:.6f}")
    
    return total_cost


def test_wide_hidden_layer():
    """
    Test network with 8 inputs -> 15 hidden -> 1 output (wide network)
    """
    print("\n\nTesting Wide Hidden Layer Network (8->15->1)")
    print("=" * 60)
    
    # Define network topology: 8 inputs -> 15 hidden -> 1 output
    layer_sizes = [8, 15, 1]
    activation_functions = [ActivationFunction.RELU, ActivationFunction.LINEAR]  # Hidden: ReLU, Output: linear
    error_function = CostFunction.MEAN_SQUARED_ERROR
    
    # Initialize the neural network
    nn = NeuralNetwork(layer_sizes, activation_functions, error_function)
    
    print(f"Network topology: {layer_sizes}")
    print(f"Number of layers (excluding input): {len(nn.layers)}")
    
    # Define weights for this topology
    weights = [
        # Hidden layer weights (15 neurons, each with bias + 8 input weights)
        [
            [0.1, 0.2, 0.3, 0.4, -0.1, 0.5, -0.2, 0.3, 0.1],      # Neuron 1
            [-0.1, 0.5, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5],    # Neuron 2
            [0.2, -0.3, 0.4, -0.1, 0.6, 0.2, -0.4, 0.1, -0.2],    # Neuron 3
            [0.3, 0.1, -0.4, 0.2, -0.3, 0.5, 0.1, 0.4, -0.1],     # Neuron 4
            [-0.2, 0.4, 0.1, -0.5, 0.2, -0.1, 0.6, -0.3, 0.2],    # Neuron 5
            [0.4, -0.2, 0.3, 0.1, -0.4, 0.3, 0.2, -0.5, 0.1],     # Neuron 6
            [0.1, 0.3, -0.1, 0.4, 0.2, -0.3, 0.5, 0.1, -0.4],     # Neuron 7
            [-0.3, 0.2, 0.5, -0.1, 0.3, 0.4, -0.2, 0.1, 0.3],     # Neuron 8
            [0.2, -0.4, 0.1, 0.3, -0.2, 0.1, 0.4, -0.3, 0.5],     # Neuron 9
            [0.5, 0.1, -0.3, 0.2, 0.4, -0.1, 0.3, 0.2, -0.4],     # Neuron 10
            [0.0, 0.3, 0.2, -0.1, 0.4, 0.1, -0.2, 0.5, 0.3],      # Neuron 11
            [-0.1, 0.2, -0.4, 0.3, 0.1, -0.3, 0.4, 0.2, -0.1],    # Neuron 12
            [0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.3, -0.2, 0.4],     # Neuron 13
            [0.2, 0.4, -0.2, 0.1, 0.3, -0.5, 0.2, 0.1, -0.3],     # Neuron 14
            [-0.4, 0.1, 0.3, -0.2, 0.5, 0.2, -0.1, 0.4, 0.1]      # Neuron 15
        ],
        # Output layer weights (1 neuron with bias + 15 input weights)
        [
            [0.1, 0.2, -0.1, 0.3, 0.1, -0.2, 0.4, -0.1, 0.2, 0.3, -0.2, 0.1, 0.4, -0.3, 0.2, 0.1]
        ]
    ]
    
    # Define input vectors (8 inputs each) and expected outputs
    input_vectors = [
        [0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.2, 0.7],
        [0.1, 0.7, 0.4, 0.8, 0.2, 0.6, 0.9, 0.3],
        [0.5, 0.2, 0.8, 0.3, 0.6, 0.1, 0.4, 0.9],
        [0.9, 0.1, 0.3, 0.7, 0.4, 0.8, 0.2, 0.5]
    ]
    
    target_outputs = [0.6, 0.4, 0.7, 0.8]
    
    print(f"\nInput vectors: {input_vectors}")
    print(f"Target outputs: {target_outputs}")
    
    # Perform forward pass and get total cost
    total_cost = nn.forward_pass(weights, input_vectors, target_outputs)
    
    print(f"\nTotal cost (MSE): {total_cost:.6f}")
    
    # Test individual predictions
    print("\nIndividual predictions:")
    print("-" * 30)
    
    for i, (input_vec, target) in enumerate(zip(input_vectors, target_outputs)):
        # Forward pass for single input
        current_inputs = input_vec
        
        for layer_idx, layer in enumerate(nn.layers):
            layer_weights = weights[layer_idx]
            current_inputs = layer.forward(current_inputs, layer_weights)
        
        prediction = current_inputs[0]
        error = (prediction - target) ** 2
        
        print(f"Input: {input_vec}, Target: {target:.3f}, Prediction: {prediction:.6f}, Error: {error:.6f}")
    
    return total_cost


if __name__ == "__main__":
    # Run tests
    cost1 = test_simple_hidden_layer()
    cost2 = test_multi_hidden_layers()
    cost3 = test_wide_hidden_layer()
    
    print(f"\n\nSummary:")
    print(f"Test 1 (8->5->1) Total Cost: {cost1:.6f}")
    print(f"Test 2 (8->10->6->3->1) Total Cost: {cost2:.6f}")
    print(f"Test 3 (8->15->1) Total Cost: {cost3:.6f}")