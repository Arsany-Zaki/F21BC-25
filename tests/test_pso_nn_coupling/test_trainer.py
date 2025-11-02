from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from pso.pso import PSOConfig
from data_prep.data_prep import DataPrep
from nn.nn_config import NNConfig
from pso.pso_config import PSOConfig
from data_prep.data_prep_config import DataPrepConfig
from configs.metadata import *

data_config = DataPrepConfig(
    norm_method = NormMethod.ZSCORE,
    norm_factors = norm_default_factors[NormMethod.ZSCORE]
)

nn_config = NNConfig(
    input_dim = 8,
    layers_sizes = [8, 1],
    activation_functions = [ActFunc.RELU, ActFunc.LINEAR],
    cost_function = CostFunc.MEAN_SQUARED_ERROR
)
pso_config = PSOConfig(
    max_iter = 10,
    swarm_size = 10,
    informant_count = 2,

    boundary_handling = BoundHandling.REFLECT,
    informant_selection = InformantSelect.SPATIAL_PROXIMITY,

    w_inertia = 0.73,
    c_personal = 1.0,
    c_social = 1.0,
    c_global = 1.0,
    jump_size = 1.0,

    dims = 8,                
    boundary_min = [],       
    boundary_max = [],       
    target_fitness = None,
)

def test_nn_trainer_using_pso_runs():
    preparator = DataPrep(data_config)
    train_df, _ = preparator.get_normalized_input_data_split()

    # Last column is target
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
