"""
Example demonstrating PSO with dimension-specific boundaries
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pso.pso import PSO, PSOConfig


def example_fitness_function(x):
    """
    Example fitness function: Rosenbrock function with offset
    This function has different optimal ranges for different dimensions
    """
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def main():
    print("PSO with Dimension-Specific Boundaries Example")
    print("=" * 60)
    
    # Example 1: Neural network weight optimization scenario
    # Different layers might have different weight ranges
    print("\nExample 1: Neural Network Weight Optimization")
    print("-" * 50)
    
    # Define bounds for different weight groups
    weight_bounds = [
        (-2.0, 2.0),    # Input layer weights: smaller range
        (-1.0, 1.0),    # Hidden layer 1 weights: very small range
        (-3.0, 3.0),    # Hidden layer 2 weights: medium range
        (-1.5, 1.5),    # Output layer weights: small range
        (-0.5, 0.5),    # Bias terms: very small range
    ]
    
    config = PSOConfig(
        dims=5,
        bounds=weight_bounds,
        swarm_size=30,
        max_iter=100,
        w_inertia=0.7,
        c_personal=1.5,
        c_social=1.5,
        c_global=1.5
    )
    
    pso = PSO(config)
    
    print("Configuration:")
    print(f"  Dimensions: {config.dims}")
    print(f"  Bounds per dimension:")
    for i, bound in enumerate(weight_bounds):
        print(f"    Dimension {i}: [{bound[0]:.1f}, {bound[1]:.1f}]")
    
    # Run optimization
    best_solution, best_fitness = pso.optimize(example_fitness_function)
    
    print(f"\nResults:")
    print(f"  Best solution: {best_solution}")
    print(f"  Best fitness: {best_fitness:.6f}")
    
    # Verify bounds are respected
    print(f"\nBounds verification:")
    for i, (sol_val, bound) in enumerate(zip(best_solution, weight_bounds)):
        within_bounds = bound[0] <= sol_val <= bound[1]
        print(f"  Dim {i}: {sol_val:.4f} in [{bound[0]:.1f}, {bound[1]:.1f}] ✓" if within_bounds else f"  Dim {i}: {sol_val:.4f} NOT in [{bound[0]:.1f}, {bound[1]:.1f}] ✗")
    
    # Example 2: Feature space with different scales
    print("\n\nExample 2: Multi-Scale Feature Optimization")
    print("-" * 50)
    
    # Different features might have vastly different natural ranges
    feature_bounds = [
        (0.0, 1.0),       # Probability/percentage (0-1)
        (-100.0, 100.0),  # Temperature in Celsius
        (0.0, 10000.0),   # Count/frequency
        (-5.0, 5.0),      # Normalized score
    ]
    
    config2 = PSOConfig(
        dims=4,
        bounds=feature_bounds,
        swarm_size=25,
        max_iter=80
    )
    
    pso2 = PSO(config2)
    
    print("Configuration:")
    print(f"  Dimensions: {config2.dims}")
    print(f"  Bounds per dimension:")
    for i, bound in enumerate(feature_bounds):
        print(f"    Feature {i}: [{bound[0]:.1f}, {bound[1]:.1f}]")
    
    # Simple quadratic fitness for this example
    def quadratic_fitness(x):
        return sum(xi**2 for xi in x)
    
    best_solution2, best_fitness2 = pso2.optimize(quadratic_fitness)
    
    print(f"\nResults:")
    print(f"  Best solution: {best_solution2}")
    print(f"  Best fitness: {best_fitness2:.6f}")
    
    # Example 3: Default bounds behavior
    print("\n\nExample 3: Default Bounds (when not specified)")
    print("-" * 50)
    
    config3 = PSOConfig(
        dims=3,
        # bounds not specified - will use default
        swarm_size=20,
        max_iter=50
    )
    
    pso3 = PSO(config3)
    
    print("Configuration:")
    print(f"  Dimensions: {config3.dims}")
    print(f"  Default bounds: {pso3.config.bounds}")
    print(f"  Bounds min: {pso3.bounds_min}")
    print(f"  Bounds max: {pso3.bounds_max}")
    
    best_solution3, best_fitness3 = pso3.optimize(quadratic_fitness)
    
    print(f"\nResults:")
    print(f"  Best solution: {best_solution3}")
    print(f"  Best fitness: {best_fitness3:.6f}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()