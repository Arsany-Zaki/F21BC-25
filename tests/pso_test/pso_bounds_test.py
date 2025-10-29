"""
Test module for PSO with dimension-specific boundaries
"""

import sys
import os
import numpy as np

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pso.pso import PSO, PSOConfig, InformantSelection, BoundaryHandling


def simple_fitness(x):
    """Simple quadratic fitness function for testing."""
    return np.sum(x**2)


def test_default_bounds():
    """Test PSO with explicitly provided bounds for all dimensions."""
    print("Testing Default Bounds")
    print("=" * 30)
    
    config = PSOConfig(
        dims=3,
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],  # Explicit bounds for 3 dimensions
        swarm_size=10,
        max_iter=50,
        informant_count=3  # Ensure valid informant count
    )
    
    pso = PSO(config)
    
    print(f"Dimensions: {pso.dims}")
    print(f"Bounds min: {pso.bounds_min}")
    print(f"Bounds max: {pso.bounds_max}")
    print(f"Bounds range: {pso.bounds_range}")
    
    # Verify default bounds are applied correctly (should be (-5.0, 5.0) for all dimensions)
    assert np.allclose(pso.bounds_min, [-5.0, -5.0, -5.0])
    assert np.allclose(pso.bounds_max, [5.0, 5.0, 5.0])
    
    # Run optimization
    best_pos, best_fit = pso.optimize(simple_fitness)
    
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fit}")
    
    # Verify solution is within bounds
    assert np.all(best_pos >= pso.bounds_min)
    assert np.all(best_pos <= pso.bounds_max)
    
    print("✓ Default bounds test passed!\n")
    return best_fit


def test_uniform_bounds_for_all_dimensions():
    """Test PSO with same bounds applied to all dimensions using list format."""
    print("Testing Uniform Bounds for All Dimensions")
    print("=" * 50)
    
    config = PSOConfig(
        dims=3,
        bounds=[(-2.0, 3.0), (-2.0, 3.0), (-2.0, 3.0)],  # Same bounds for all dimensions
        swarm_size=10,
        max_iter=50,
        informant_count=3  # Ensure valid informant count
    )
    
    pso = PSO(config)
    
    print(f"Dimensions: {pso.dims}")
    print(f"Bounds min: {pso.bounds_min}")
    print(f"Bounds max: {pso.bounds_max}")
    print(f"Bounds range: {pso.bounds_range}")
    
    # Verify bounds are applied correctly
    assert np.allclose(pso.bounds_min, [-2.0, -2.0, -2.0])
    assert np.allclose(pso.bounds_max, [3.0, 3.0, 3.0])
    
    # Run optimization
    best_pos, best_fit = pso.optimize(simple_fitness)
    
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fit}")
    
    # Verify solution is within bounds
    assert np.all(best_pos >= pso.bounds_min)
    assert np.all(best_pos <= pso.bounds_max)
    
    print("✓ Uniform bounds test passed!\n")
    return best_fit


def test_dimension_specific_bounds():
    """Test PSO with different bounds for each dimension."""
    print("Testing Dimension-Specific Bounds")
    print("=" * 50)
    
    # Different bounds for each dimension
    dimension_bounds = [
        (-1.0, 1.0),    # Dimension 0: [-1, 1]
        (-5.0, 0.0),    # Dimension 1: [-5, 0]
        (2.0, 8.0),     # Dimension 2: [2, 8]
        (-10.0, 10.0)   # Dimension 3: [-10, 10]
    ]
    
    config = PSOConfig(
        dims=4,
        bounds=dimension_bounds,
        swarm_size=15,
        max_iter=100
    )
    
    pso = PSO(config)
    
    print(f"Dimensions: {pso.dims}")
    print(f"Bounds min: {pso.bounds_min}")
    print(f"Bounds max: {pso.bounds_max}")
    print(f"Bounds range: {pso.bounds_range}")
    
    # Verify bounds are applied correctly
    expected_min = np.array([-1.0, -5.0, 2.0, -10.0])
    expected_max = np.array([1.0, 0.0, 8.0, 10.0])
    
    assert np.allclose(pso.bounds_min, expected_min)
    assert np.allclose(pso.bounds_max, expected_max)
    
    # Run optimization
    best_pos, best_fit = pso.optimize(simple_fitness)
    
    print(f"Best position: {best_pos}")
    print(f"Best fitness: {best_fit}")
    
    # Verify solution is within bounds for each dimension
    for i in range(pso.dims):
        assert best_pos[i] >= pso.bounds_min[i], f"Dimension {i}: {best_pos[i]} < {pso.bounds_min[i]}"
        assert best_pos[i] <= pso.bounds_max[i], f"Dimension {i}: {best_pos[i]} > {pso.bounds_max[i]}"
    
    print("✓ Dimension-specific bounds test passed!\n")
    return best_fit


def test_boundary_handling_with_dimension_bounds():
    """Test boundary handling strategies with dimension-specific bounds."""
    print("Testing Boundary Handling with Dimension-Specific Bounds")
    print("=" * 60)
    
    dimension_bounds = [
        (-2.0, 2.0),    # Dimension 0
        (-1.0, 3.0),    # Dimension 1
        (0.0, 5.0)      # Dimension 2
    ]
    
    # Test with CLIP boundary handling
    config_clip = PSOConfig(
        dims=3,
        bounds=dimension_bounds,
        swarm_size=20,
        max_iter=50,
        boundary_handling=BoundaryHandling.CLIP
    )
    
    pso_clip = PSO(config_clip)
    best_pos_clip, best_fit_clip = pso_clip.optimize(simple_fitness)
    
    print(f"CLIP - Best position: {best_pos_clip}")
    print(f"CLIP - Best fitness: {best_fit_clip}")
    
    # Test with REFLECT boundary handling
    config_reflect = PSOConfig(
        dims=3,
        bounds=dimension_bounds,
        swarm_size=20,
        max_iter=50,
        boundary_handling=BoundaryHandling.REFLECT
    )
    
    pso_reflect = PSO(config_reflect)
    best_pos_reflect, best_fit_reflect = pso_reflect.optimize(simple_fitness)
    
    print(f"REFLECT - Best position: {best_pos_reflect}")
    print(f"REFLECT - Best fitness: {best_fit_reflect}")
    
    # Verify both solutions are within bounds
    for i in range(3):
        assert best_pos_clip[i] >= dimension_bounds[i][0]
        assert best_pos_clip[i] <= dimension_bounds[i][1]
        assert best_pos_reflect[i] >= dimension_bounds[i][0]
        assert best_pos_reflect[i] <= dimension_bounds[i][1]
    
    print("✓ Boundary handling test passed!\n")
    return best_fit_clip, best_fit_reflect


def test_swarm_initialization_with_dimension_bounds():
    """Test that swarm is properly initialized within dimension-specific bounds."""
    print("Testing Swarm Initialization with Dimension-Specific Bounds")
    print("=" * 60)
    
    dimension_bounds = [
        (-3.0, -1.0),   # Dimension 0: tight negative range
        (5.0, 7.0),     # Dimension 1: tight positive range
        (-10.0, 15.0)   # Dimension 2: wide range
    ]
    
    config = PSOConfig(
        dims=3,
        bounds=dimension_bounds,
        swarm_size=50,
        max_iter=1  # Just test initialization
    )
    
    pso = PSO(config)
    
    print(f"Swarm size: {config.swarm_size}")
    print(f"Dimension bounds: {dimension_bounds}")
    
    # Check that all particles are initialized within bounds
    for i in range(config.swarm_size):
        for dim in range(3):
            pos = pso.positions[i, dim]
            min_bound, max_bound = dimension_bounds[dim]
            assert min_bound <= pos <= max_bound, f"Particle {i}, dim {dim}: {pos} not in [{min_bound}, {max_bound}]"
    
    print(f"✓ All {config.swarm_size} particles initialized within bounds!")
    
    # Show some sample positions
    print("\nSample particle positions:")
    for i in range(min(5, config.swarm_size)):
        print(f"  Particle {i}: {pso.positions[i]}")
    
    print("✓ Swarm initialization test passed!\n")


def test_valid_configurations():
    """Test that all valid configurations work correctly."""
    print("Testing Valid Configurations")
    print("=" * 30)
    
    # Test with proper bounds matching dimensions
    try:
        config = PSOConfig(
            dims=3,
            bounds=[(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]  # Proper bounds for 3 dimensions
        )
        pso = PSO(config)
        assert len(pso.bounds) == 3
        assert pso.bounds[0] == (-1.0, 1.0)
        assert pso.bounds[1] == (-2.0, 2.0)
        assert pso.bounds[2] == (-3.0, 3.0)
        print("✓ Proper bounds configuration works correctly")
    except Exception as e:
        assert False, f"Unexpected error: {e}"
    
    # Test with uniform bounds (list format)
    try:
        config = PSOConfig(
            dims=2,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)]  # Uniform bounds in list format
        )
        pso = PSO(config)
        assert len(pso.bounds) == 2
        assert all(bound == (-1.0, 1.0) for bound in pso.bounds)
        print("✓ Uniform bounds in list format work correctly")
    except Exception as e:
        assert False, f"Unexpected error: {e}"
    
    print("✓ Valid configurations test passed!\n")


if __name__ == "__main__":
    print("PSO Dimension-Specific Boundaries Test Suite")
    print("=" * 70)
    
    # Run all tests
    fit1 = test_default_bounds()
    fit2 = test_uniform_bounds_for_all_dimensions()
    fit3 = test_dimension_specific_bounds()
    fit4, fit5 = test_boundary_handling_with_dimension_bounds()
    test_swarm_initialization_with_dimension_bounds()
    test_valid_configurations()
    
    print("Summary:")
    print(f"Default bounds fitness: {fit1:.6f}")
    print(f"Uniform bounds fitness: {fit2:.6f}")
    print(f"Dimension-specific bounds fitness: {fit3:.6f}")
    print(f"CLIP boundary handling fitness: {fit4:.6f}")
    print(f"REFLECT boundary handling fitness: {fit5:.6f}")
    print("\n✓ All tests passed successfully!")