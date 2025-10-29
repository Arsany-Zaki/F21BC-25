import sys
import os
import numpy as np
import pytest
# Add project root to sys.path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.data_config import data as data_cfg
from config.output_config import output as output_cfg
from config.activation_functions_enum import ActivationFunction
from config.cost_functions_enum import CostFunction
from config.boundary_handling_enum import BoundaryHandling
from config.informant_selection_enum import InformantSelection
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO

from pso.pso import PSOParams
from input_data_Processor.prepare_input_data import DataPreparator

# Use NNConfig from generic_nn and PSOConfig from pso module
from nn.generic_nn import NNConfig
from pso.pso import PSOParams as PSOConfig
from dataclasses import dataclass

@dataclass
class TestCaseConfig:
    nn: NNConfig
    pso: PSOConfig
    # add 'data' or other sections as needed

test_case_config = TestCaseConfig(
    nn=NNConfig(
        layer_sizes=[8, 8, 1],
        activation_functions=[ActivationFunction.TANH, ActivationFunction.LINEAR],
        error_function=CostFunction.MEAN_SQUARED_ERROR
    ),
    pso=PSOConfig(
        max_iter=20,
        swarm_size=5,
        informant_count=2,
        
        boundary_handling=BoundaryHandling.REFLECT,
        informant_selection=InformantSelection.STATIC_RANDOM,

        w_inertia=0.9,
        c_personal=1.4,
        c_social=1.4,
        c_global=1.4,
        jump_size=1.0,
        vel_limit=0.9,

        dims=0,                #to be set dynamically,
        bounds=[],             #to be set dynamically,
        target_fitness=None,   #not required,
    )
)

def test_nn_trainer_using_pso_runs():

    # Load config from Python files (nested structure)
    config = {"data": data_cfg, "output": output_cfg}

    # Prepare data
    preparator = DataPreparator(config)
    preparator.prepare_data_for_training()
    train_df, _ = preparator.get_normalized_input_data_split()

    # Assume last column is target
    X = train_df.iloc[:, :-1].values.tolist()
    y = train_df.iloc[:, -1].values.tolist()
    training_data = {'inputs': X, 'targets': y}
    
    # Dynamically set PSOConfig.dims and bounds based on NN topology and activations
    # Create a temporary trainer to access the boundary calculation
    temp_trainer = NNTrainerUsingPSO(training_data, {'pso_params': test_case_config.pso, 'nn_params': test_case_config.nn})
    boundaries = temp_trainer._calculate_pso_feature_boundaries()
    test_case_config.pso.dims = len(boundaries)
    test_case_config.pso.bounds = boundaries

    trainer = NNTrainerUsingPSO(training_data, {'pso_params': test_case_config.pso, 'nn_params': test_case_config.nn})
    best_weights, best_fitness = trainer.train_nn()
    print(f"Best Fitness: {best_fitness}")
    

if __name__ == "__main__":
    test_nn_trainer_using_pso_runs()
