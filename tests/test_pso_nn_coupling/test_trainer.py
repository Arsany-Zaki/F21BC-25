from settings.data_settings import data as data_cfg
from settings.output_settings import output as output_cfg
from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from pso.pso import PSOConfig
from input_data_Processor.input_data_processor import DataPreparator
from nn.nn import NNConfig
from pso.pso import PSOConfig as PSOConfig
from dataclasses import dataclass
from settings.enumerations import ActivationFunction as act_func, CostFunction as cost_func, BoundaryHandling as bound_handling, InformantSelection as informant_selec

nn_config = NNConfig(
    input_dim = 8,
    layers_sizes = [16, 8, 1],
    activation_functions = [act_func.RELU, act_func.RELU, act_func.LINEAR],
    cost_function = cost_func.MEAN_SQUARED_ERROR
)
pso_config = PSOConfig(
    max_iter = 500,
    swarm_size = 20,
    informant_count = 10,

    boundary_handling = bound_handling.REFLECT,
    informant_selection = informant_selec.STATIC_RANDOM,

    w_inertia = 1,
    c_personal = 1.0,
    c_social = 1.4,
    c_global = 1.4,
    jump_size = 1.4,

    dims = 8,                
    boundary_min = [],       
    boundary_max = [],       
    target_fitness = None,
)


def test_nn_trainer_using_pso_runs():

    # Load config from Python files (nested structure)
    config = {"data": data_cfg, "output": output_cfg}

    # Prepare data
    preparator = DataPreparator(config)
    train_df, _ = preparator.get_normalized_input_data_split()

    # Assume last column is target
    X = train_df.iloc[:, :-1].values.tolist()
    y = train_df.iloc[:, -1].values.tolist()
    training_data = {'inputs': X, 'targets': y}
    
    # Dynamically set PSOConfig.dims and boundary_min/boundary_max based on NN topology and activations
    temp_trainer = NNTrainerUsingPSO(training_data, pso_config, nn_config)
    boundaries = temp_trainer._calculate_pso_feature_boundaries()
    pso_config.dims = len(boundaries)
    pso_config.boundary_min = [b[0] for b in boundaries]
    pso_config.boundary_max = [b[1] for b in boundaries]

    trainer = NNTrainerUsingPSO(training_data, pso_config, nn_config)
    _, bestf_custom_pso = trainer.train_nn_using_pso()
    #_, bestf_pyswarm_pso = trainer.train_nn_using_pyswarm_pso()
    #_, bestf_pyswarm_pso_default = trainer.train_nn_pyswarm_pso_default()
    
    print(f"Best Fitness of CUSTOM PSO : {bestf_custom_pso}")
    #print(f"Best Fitness of PYSWARM PSO : {bestf_pyswarm_pso}")
    #print(f"Best Fitness of PYSWARM PSO DEFAULT PARAMS: {bestf_pyswarm_pso_default}")

if __name__ == "__main__":
    test_nn_trainer_using_pso_runs()
