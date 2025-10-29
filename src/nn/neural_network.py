"""
Neural Network Forward Pass Implementation
Contains classes for NeuralNetwork, Layer, and Neuron
"""

import numpy as np
import math
from typing import List, Callable, Any
from enum import Enum


class ActivationFunction(Enum):
    """Enumeration for activation functions"""
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "linear"


class CostFunction(Enum):
    """Enumeration for cost functions"""
    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae"


class Neuron:
    """
    Represents a single neuron in a neural network layer
    """
    
    def __init__(self, activation_function: ActivationFunction):
        """
        Initialize a neuron with an activation function
        
        Args:
            activation_function: ActivationFunction enum value
        """
        self.activation_function = activation_function
        self.output = 0.0
        self.weighted_sum = 0.0
    
    def forward(self, inputs: List[float], weights: List[float]) -> float:
        """
        Perform forward pass for this neuron
        
        Args:
            inputs: List of input values
            weights: List of weights (first weight is bias, rest are input weights)
            
        Returns:
            Activated output of the neuron
        """
        # First weight is bias, rest are input weights
        bias = weights[0]
        input_weights = weights[1:]
        
        # Calculate weighted sum: bias + sum(input_i * weight_i)
        self.weighted_sum = bias + sum(inp * w for inp, w in zip(inputs, input_weights))
        
        # Apply activation function
        self.output = self._apply_activation(self.weighted_sum)
        
        return self.output
    
    def _apply_activation(self, x: float) -> float:
        """Apply the activation function to the input"""
        if self.activation_function == ActivationFunction.SIGMOID:
            return 1.0 / (1.0 + math.exp(-np.clip(x, -500, 500)))
        elif self.activation_function == ActivationFunction.RELU:
            return max(0.0, x)
        elif self.activation_function == ActivationFunction.TANH:
            return math.tanh(x)
        elif self.activation_function == ActivationFunction.LINEAR:
            return x
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")


class Layer:
    """
    Represents a layer of neurons in a neural network
    """
    
    def __init__(self, num_neurons: int, activation_function: ActivationFunction):
        """
        Initialize a layer with specified number of neurons
        
        Args:
            num_neurons: Number of neurons in this layer
            activation_function: ActivationFunction enum value for all neurons in this layer
        """
        self.neurons = [Neuron(activation_function) for _ in range(num_neurons)]
        self.outputs = []
    
    def forward(self, inputs: List[float], weights: List[List[float]]) -> List[float]:
        """
        Perform forward pass for this layer
        
        Args:
            inputs: List of input values to the layer
            weights: List of weight lists, one for each neuron
            
        Returns:
            List of outputs from all neurons in the layer
        """
        self.outputs = []
        
        for i, neuron in enumerate(self.neurons):
            output = neuron.forward(inputs, weights[i])
            self.outputs.append(output)
        
        return self.outputs


class NeuralNetwork:
    """
    Neural Network class that performs forward pass with given weights
    """
    
    def __init__(self, 
                 layer_sizes: List[int], 
                 activation_functions: List[ActivationFunction], 
                 error_function: CostFunction):
        """
        Initialize neural network structure
        
        Args:
            layer_sizes: List of integers specifying number of neurons in each layer
            activation_functions: List of ActivationFunction enums for each layer
            error_function: CostFunction enum to calculate error between predicted and actual outputs
        """
        self.layer_sizes = layer_sizes
        self.error_function = error_function
        self.layers = []
        
        # Create layers (excluding input layer which is just pass-through)
        for i in range(1, len(layer_sizes)):
            layer = Layer(layer_sizes[i], activation_functions[i-1])
            self.layers.append(layer)
    
    def forward_pass(self, 
                    weights: List[List[List[float]]], 
                    input_vectors: List[List[float]], 
                    target_outputs: List[float]) -> float:
        """
        Perform forward pass and calculate total cost
        
        Args:
            weights: 3D list structure: [layer][neuron][weight] where first weight is bias
            input_vectors: List of input vectors to process
            target_outputs: List of expected outputs for each input vector
            
        Returns:
            Total cost across all input vectors
        """
        total_cost = 0.0
        predictions = []
        
        for input_vector, target_output in zip(input_vectors, target_outputs):
            # Forward pass through the network
            current_inputs = input_vector
            
            for layer_idx, layer in enumerate(self.layers):
                layer_weights = weights[layer_idx]
                current_inputs = layer.forward(current_inputs, layer_weights)
            
            # Get prediction (output of last layer)
            prediction = current_inputs[0] if len(current_inputs) == 1 else current_inputs
            predictions.append(prediction)
            
            # Calculate cost for this sample
            if isinstance(prediction, list):
                cost = self._apply_cost_function(prediction, [target_output])
            else:
                cost = self._apply_cost_function([prediction], [target_output])
            
            total_cost += cost
        
        return total_cost
    
    def _apply_cost_function(self, predictions: List[float], targets: List[float]) -> float:
        """Apply the cost function to predictions and targets"""
        if self.error_function == CostFunction.MEAN_SQUARED_ERROR:
            return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
        elif self.error_function == CostFunction.MEAN_ABSOLUTE_ERROR:
            return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
        else:
            raise ValueError(f"Unknown cost function: {self.error_function}")

