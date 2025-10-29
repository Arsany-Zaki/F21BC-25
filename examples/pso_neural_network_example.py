"""
Example: Using PSO to optimize neural network weights.
Demonstrates integration between DeepNN and PSO classes.
"""

import numpy as np
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ann.deep_nn import DeepNN, NetworkConfig
from pso.pso import PSO, PSOConfig


def create_sample_dataset():
    """Create a simple dataset for neural network training."""
    # Simple function: f(x1, x2) = x1 + x2 (with some noise)
    np.random.seed(42)
    
    num_samples = 50
    input_data = np.random.uniform(-1, 1, (num_samples, 2))
    target_output = np.sum(input_data, axis=1, keepdims=True) + np.random.normal(0, 0.1, (num_samples, 1))
    
    return input_data, target_output


def pso_neural_network_optimization():
    """Demonstrate PSO optimization of neural network weights."""
    print("🧠🐝 PSO NEURAL NETWORK OPTIMIZATION")
    print("=" * 60)
    
    # Create dataset
    input_data, target_output = create_sample_dataset()
    print(f"Dataset: {input_data.shape[0]} samples, {input_data.shape[1]} inputs, {target_output.shape[1]} outputs")
    
    # Create neural network
    network_config = NetworkConfig(
        num_inputs=2,
        hidden_layers=[4, 3],
        num_outputs=1,
        activation_functions=["tanh", "tanh", "linear"],
        error_function="mse"
    )
    
    network = DeepNN(network_config)
    network_info = network.get_network_info()
    
    print(f"Network: {network_info['architecture']}")
    print(f"Total weights to optimize: {network_info['total_weights']}")
    print(f"Activation functions: {network_info['activation_functions']}")
    
    # Define fitness function for PSO
    def fitness_function(weights):
        """PSO fitness function - returns neural network cost to minimize."""
        return network.forward_pass(weights, input_data, target_output)
    
    # Test initial random weights  
    np.random.seed(123)
    initial_weights = np.random.uniform(-1, 1, network_info['total_weights'])
    initial_cost = fitness_function(initial_weights)
    print(f"Initial random cost: {initial_cost:.6f}")
    
    # Configure PSO
    weight_bounds = (-2.0, 2.0)  # Weight bounds for neural network
    
    pso_config = PSOConfig(
        swarm_size=30,
        dims=network_info['total_weights'],
        bounds=weight_bounds,
        max_iter=100,
        inertia=0.9,
        pbest=1.4,
        sbest=1.4,
        gbest=1.4,
        informant_selection="static_random",
        informant_count=3,
        boundary="clip"
    )
    
    print(f"\nPSO Configuration:")
    print(f"  Swarm size: {pso_config.swarm_size}")
    print(f"  Dimensions: {pso_config.dims}")
    print(f"  Max iterations: {pso_config.max_iter}")
    print(f"  Weight bounds: {pso_config.bounds}")
    
    # Run PSO optimization
    print(f"\n🚀 Starting PSO optimization...")
    pso = PSO(pso_config)
    
    best_weights, best_cost = pso.optimize(fitness_function)
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"  Initial cost: {initial_cost:.6f}")
    print(f"  Final cost:   {best_cost:.6f}")
    print(f"  Improvement:  {initial_cost - best_cost:.6f} ({((initial_cost - best_cost) / initial_cost * 100):.2f}%)")
    
    # Test the optimized network
    print(f"\n🧪 Testing optimized network:")
    test_inputs = np.array([
        [0.5, 0.3],   # Should output ≈ 0.8
        [-0.2, 0.7],  # Should output ≈ 0.5
        [0.0, 0.0],   # Should output ≈ 0.0
        [1.0, -1.0]   # Should output ≈ 0.0
    ])
    
    # Use the network with optimized weights (we need to simulate the forward pass for prediction)
    # For simplicity, we'll just show that the fitness function works with these test cases
    for i, test_input in enumerate(test_inputs):
        expected_output = np.sum(test_input)
        test_cost = network.forward_pass(best_weights, test_input.reshape(1, -1), 
                                       np.array([[expected_output]]))
        print(f"  Input: {test_input} → Expected: {expected_output:.3f}, Cost: {test_cost:.6f}")
    
    return best_weights, best_cost, network


if __name__ == "__main__":
    # Run the PSO neural network optimization example
    best_weights, best_cost, network = pso_neural_network_optimization()
    
    print("\n" + "=" * 60)
    print("✅ PSO Neural Network optimization completed!")
    print("🎯 The neural network weights have been optimized using PSO")
    print("🔗 This demonstrates the integration between DeepNN and PSO classes")
    print("=" * 60)