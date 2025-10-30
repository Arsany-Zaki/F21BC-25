import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from settings.data_settings import data as data_cfg
from settings.output_settings import output as output_cfg
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from pso.pso import PSOConfig
from input_data_Processor.prepare_input_data import DataPreparator
from nn.nn import NNConfig
from pso.pso import PSOConfig as PSOConfig
from dataclasses import dataclass
from settings.enumerations import *


@dataclass
class TestCaseConfig:
    nn: NNConfig
    pso: PSOConfig
    # add 'data' or other sections as needed

test_case_config = TestCaseConfig(
    nn=NNConfig(
        layer_sizes=[8, 8, 1],
        activation_functions=[ActivationFunction.RELU, ActivationFunction.LINEAR],
        error_function=CostFunction.MEAN_SQUARED_ERROR
    ),
    pso=PSOConfig(
        max_iter=50,
        swarm_size=10,
        informant_count=2,

        boundary_handling=BoundaryHandling.REFLECT,
        informant_selection=InformantSelection.STATIC_RANDOM,

        w_inertia=0.5,
        c_personal=0.5,
        c_social=0.0,
        c_global=0.5,
        jump_size=1.0,

        dims=8,                # to be set dynamically
        boundary_min=[],       # to be set dynamically
        boundary_max=[],       # to be set dynamically
        target_fitness=None,   # not required
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
    
    # Dynamically set PSOConfig.dims and boundary_min/boundary_max based on NN topology and activations
    temp_trainer = NNTrainerUsingPSO(training_data, {'pso_params': test_case_config.pso, 'nn_params': test_case_config.nn})
    boundaries = temp_trainer._calculate_pso_feature_boundaries()
    test_case_config.pso.dims = len(boundaries)
    test_case_config.pso.boundary_min = [b[0] for b in boundaries]
    test_case_config.pso.boundary_max = [b[1] for b in boundaries]

    trainer = NNTrainerUsingPSO(training_data, {'pso_params': test_case_config.pso, 'nn_params': test_case_config.nn})
    _, bestf_custom_pso = trainer.train_nn()
    _, bestf_pyswarm_pso = trainer.train_nn_pyswarm_pso()
    #_, bestf_pyswarm_pso_default = trainer.train_nn_pyswarm_pso_default()
    
    print(f"Best Fitness of CUSTOM PSO : {bestf_custom_pso}")
    print(f"Best Fitness of PYSWARM PSO : {bestf_pyswarm_pso}")
    #print(f"Best Fitness of PYSWARM PSO DEFAULT PARAMS: {bestf_pyswarm_pso_default}")

if __name__ == "__main__":
    test_nn_trainer_using_pso_runs()
