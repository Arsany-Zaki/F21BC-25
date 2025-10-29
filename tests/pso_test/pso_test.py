#!/usr/bin/env python3
"""
PSO Parameter Testing and Statistical Analysis
Tests different parameter configurations and provides statistical summaries.
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pso.pso import PSO, PSOConfig, InformantSelection, BoundaryHandling


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function - global minimum at [1, 1] with value 0."""
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def sphere(x: np.ndarray) -> float:
    """Sphere function - global minimum at [0, 0] with value 0."""
    return np.sum(x**2)


def ackley(x: np.ndarray) -> float:
    """Ackley function - global minimum at [0, 0] with value 0."""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - 
            np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def distance_to_optimum(position: np.ndarray, optimum: np.ndarray) -> float:
    """Calculate Euclidean distance to global optimum."""
    return np.linalg.norm(position - optimum)


def run_pso_experiment(config: PSOConfig, n_runs: int = 10) -> Dict:
    """Run PSO multiple times with given configuration and collect statistics."""
    results = {
        'fitness_values': [],
        'distances': [],
        'execution_times': []
    }
    
    optimum = np.array([0.0, 0.0])  # Sphere function optimum
    
    for run in range(n_runs):
        pso = PSO(config)
        
        start_time = time.time()
        best_pos, best_fit = pso.optimize(sphere)
        end_time = time.time()
        
        distance = distance_to_optimum(best_pos, optimum)
        execution_time = end_time - start_time
        
        results['fitness_values'].append(best_fit)
        results['distances'].append(distance)
        results['execution_times'].append(execution_time)
    
    # Calculate statistics
    fitness_array = np.array(results['fitness_values'])
    distance_array = np.array(results['distances'])
    time_array = np.array(results['execution_times'])
    
    return {
        'fitness_mean': np.mean(fitness_array),
        'fitness_std': np.std(fitness_array),
        'fitness_min': np.min(fitness_array),
        'fitness_max': np.max(fitness_array),
        'distance_mean': np.mean(distance_array),
        'distance_std': np.std(distance_array),
        'distance_min': np.min(distance_array),
        'distance_max': np.max(distance_array),
        'time_mean': np.mean(time_array),
        'time_std': np.std(time_array),
        'n_runs': n_runs
    }


def create_parameter_configurations() -> List[Tuple[str, PSOConfig]]:
    """Create different parameter configurations to test."""
    configurations = []
    
    # Base configuration
    base_config = PSOConfig(
        swarm_size=20,
        dims=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        max_iter=50,
        target_fitness=1e-4
    )
    
    # Test different informant selections
    for selection in InformantSelection:
        config = PSOConfig(
            swarm_size=base_config.swarm_size,
            dims=base_config.dims,
            bounds=base_config.bounds,
            max_iter=base_config.max_iter,
            target_fitness=base_config.target_fitness,
            informant_selection=selection,
            informant_count=min(3, base_config.swarm_size - 1),  # Ensure valid informant count
            w_inertia=0.9,
            c_personal=1.4,
            c_social=1.4,
            c_global=1.4
        )
        configurations.append((f"{selection.name.lower()}", config))
    
    # Test different coefficient combinations
    coefficient_sets = [
        ("low_coeff", {"w_inertia": 0.5, "c_personal": 1.0, "c_social": 1.0, "c_global": 1.0}),
        ("high_coeff", {"w_inertia": 0.9, "c_personal": 2.0, "c_social": 2.0, "c_global": 2.0}),
        ("explore", {"w_inertia": 0.9, "c_personal": 0.5, "c_social": 2.0, "c_global": 2.0}),
        ("exploit", {"w_inertia": 0.4, "c_personal": 2.0, "c_social": 0.5, "c_global": 0.5}),
        ("balanced", {"w_inertia": 0.7, "c_personal": 1.5, "c_social": 1.5, "c_global": 1.5}),
        ("low_w", {"w_inertia": 0.3, "c_personal": 1.4, "c_social": 1.4, "c_global": 1.4}),
        ("mid_w", {"w_inertia": 0.6, "c_personal": 1.4, "c_social": 1.4, "c_global": 1.4}),
        ("high_w", {"w_inertia": 0.95, "c_personal": 1.4, "c_social": 1.4, "c_global": 1.4}),
        ("personal_bias", {"w_inertia": 0.7, "c_personal": 2.5, "c_social": 1.0, "c_global": 1.0}),
        ("social_bias", {"w_inertia": 0.7, "c_personal": 1.0, "c_social": 2.5, "c_global": 1.0}),
        ("global_bias", {"w_inertia": 0.7, "c_personal": 1.0, "c_social": 1.0, "c_global": 2.5}),
        ("conservative", {"w_inertia": 0.8, "c_personal": 0.8, "c_social": 0.8, "c_global": 0.8}),
        ("aggressive", {"w_inertia": 0.4, "c_personal": 2.5, "c_social": 2.5, "c_global": 2.5}),
        ("classic", {"w_inertia": 0.729, "c_personal": 1.494, "c_social": 1.494, "c_global": 1.494}),
        ("minimal", {"w_inertia": 0.1, "c_personal": 0.5, "c_social": 0.5, "c_global": 0.5}),
    ]
    
    for name, coeffs in coefficient_sets:
        config = PSOConfig(
            swarm_size=base_config.swarm_size,
            dims=base_config.dims,
            bounds=base_config.bounds,
            max_iter=base_config.max_iter,
            target_fitness=base_config.target_fitness,
            informant_selection=InformantSelection.STATIC_RANDOM,
            informant_count=min(3, base_config.swarm_size - 1),  # Ensure valid informant count
            **coeffs
        )
        configurations.append((name, config))
    
    # Test different swarm sizes
    for size in [10, 15, 20, 25, 30, 40]:
        config = PSOConfig(
            swarm_size=size,
            dims=base_config.dims,
            bounds=base_config.bounds,
            max_iter=base_config.max_iter,
            target_fitness=base_config.target_fitness,
            informant_selection=InformantSelection.STATIC_RANDOM,
            informant_count=min(3, size - 1),  # Ensure valid informant count for each swarm size
            w_inertia=0.9,
            c_personal=1.4,
            c_social=1.4,
            c_global=1.4
        )
        configurations.append((f"size_{size}", config))
    
    return configurations


def print_results_table(results: List[Tuple[str, Dict, PSOConfig]]):
    """Print formatted results table."""
    print("\n" + "="*90)
    print("PSO PARAMETER CONFIGURATION ANALYSIS")
    print("="*90)
    print("Problem: Sphere Function Optimization (Target: [0, 0], Value: 0)")
    print("Runs per configuration: 10")
    print("="*90)
    
    # Header
    header = (
        f"{'#':<3}  "
        f"{'Fitness (Mean ± Std)':<18}  "
        f"{'Distance (Mean ± Std)':<18}  "
        f"{'w    cp   cs   cg':<14}  "
        f"{'Informant':<12}  "
        f"{'Size':<6}"
    )
    print(header)
    print("-" * 90)
    
    # Sort results by distance (mean) in ascending order (best first)
    results.sort(key=lambda x: x[1]['distance_mean'])
    
    # ANSI color codes
    LIGHT_GRAY_BG = '\033[48;5;250m'  # Light gray background
    RESET = '\033[0m'                 # Reset color
    
    for i, (config_name, stats, config) in enumerate(results):
        # Format fitness (with spaces around ±)
        fitness_str = f"{stats['fitness_mean']:.5f} ± {stats['fitness_std']:.5f}"
        
        # Format distance (with spaces around ±)
        distance_str = f"{stats['distance_mean']:.3f} ± {stats['distance_std']:.3f}"
        
        # Format coefficients (with more spacing)
        coefficients_str = f"{config.w_inertia:.1f}   {config.c_personal:.1f}   {config.c_social:.1f}   {config.c_global:.1f}"
        
        # Format informant selection (abbreviated)
        informant_str = config.informant_selection.name.lower()[:11]
        
        # Format swarm size
        swarm_str = f"{config.swarm_size}"
        
        row = (
            f"{i+1:<3}  "
            f"{fitness_str:<18}  "
            f"{distance_str:<18}  "
            f"{coefficients_str:<14}  "
            f"{informant_str:<12}  "
            f"{swarm_str:<6}"
        )
        
        # Apply light gray background to every other row (even indices)
        if i % 2 == 0:
            print(f"{LIGHT_GRAY_BG}{row}{RESET}")
        else:
            print(row)
    
    print("-" * 90)
    
    # Summary statistics
    print("\nSUMMARY:")
    best_config = results[0]
    worst_config = results[-1]
    
    print(f"🏆 Best Configuration:  {best_config[0]}")
    print(f"   Average Fitness:     {best_config[1]['fitness_mean']:.6f} ± {best_config[1]['fitness_std']:.6f}")
    print(f"   Average Distance:    {best_config[1]['distance_mean']:.4f} ± {best_config[1]['distance_std']:.4f}")
    print(f"   Best Single Run:     {best_config[1]['fitness_min']:.6f}")
    
    print(f"\n📉 Worst Configuration: {worst_config[0]}")
    print(f"   Average Fitness:     {worst_config[1]['fitness_mean']:.6f} ± {worst_config[1]['fitness_std']:.6f}")
    print(f"   Average Distance:    {worst_config[1]['distance_mean']:.4f} ± {worst_config[1]['distance_std']:.4f}")
    
    # Find best single run across all configurations
    best_single_run = min(results, key=lambda x: x[1]['fitness_min'])
    print(f"\n🎯 Best Single Run:     {best_single_run[1]['fitness_min']:.6f} ({best_single_run[0]})")
    
    print("="*110)


def main():
    """Main function to run parameter testing."""
    print("Starting PSO Parameter Configuration Testing...")
    
    # Create parameter configurations
    configurations = create_parameter_configurations()
    
    print(f"Testing {len(configurations)} different configurations...")
    print("Each configuration will be run 10 times for statistical analysis.")
    
    # Run experiments
    results = []
    for i, (config_name, config) in enumerate(configurations, 1):
        print(f"\nRunning configuration {i}/{len(configurations)}: {config_name}")
        stats = run_pso_experiment(config, n_runs=5)
        results.append((config_name, stats, config))
        print(f"  Completed - Average fitness: {stats['fitness_mean']:.6f}")
    
    # Print results table
    print_results_table(results)


if __name__ == "__main__":
    main()
