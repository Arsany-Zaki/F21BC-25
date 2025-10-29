"""
Quick test for PSO with dimension-specific boundaries
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pso.pso import PSO, PSOConfig


def simple_fitness(x):
    """Simple quadratic fitness function."""
    return np.sum(x**2)


def test_pso_bounds():
    print("Testing PSO with Dimension-Specific Boundaries")
    print("=" * 50)
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration:")
    config1 = PSOConfig()
    pso1 = PSO(config1)
    
    print(f"   Dims: {pso1.dims}")
    print(f"   Bounds: {pso1.bounds}")
    print(f"   Bounds min: {pso1.bounds_min}")
    print(f"   Bounds max: {pso1.bounds_max}")
    
    best_pos1, best_fit1 = pso1.optimize(simple_fitness)
    print(f"   Best position: {best_pos1}")
    print(f"   Best fitness: {best_fit1:.6f}")
    
    # Test 2: Custom dimension-specific bounds
    print("\n2. Testing dimension-specific bounds:")
    custom_bounds = [(-1.0, 1.0), (-2.0, 3.0), (0.0, 5.0)]
    config2 = PSOConfig(
        dims=3,
        bounds=custom_bounds,
        swarm_size=20,
        max_iter=50
    )
    
    pso2 = PSO(config2)
    
    print(f"   Dims: {pso2.dims}")
    print(f"   Bounds: {pso2.bounds}")
    print(f"   Bounds min: {pso2.bounds_min}")
    print(f"   Bounds max: {pso2.bounds_max}")
    
    best_pos2, best_fit2 = pso2.optimize(simple_fitness)
    print(f"   Best position: {best_pos2}")
    print(f"   Best fitness: {best_fit2:.6f}")
    
    # Verify bounds are respected
    print("\n3. Verifying bounds are respected:")
    for i, (pos_val, bound) in enumerate(zip(best_pos2, custom_bounds)):
        within_bounds = bound[0] <= pos_val <= bound[1]
        status = "✓" if within_bounds else "✗"
        print(f"   Dim {i}: {pos_val:.4f} in [{bound[0]:.1f}, {bound[1]:.1f}] {status}")
    
    print("\n✓ All tests completed successfully!")
    return best_fit1, best_fit2


if __name__ == "__main__":
    test_pso_bounds()