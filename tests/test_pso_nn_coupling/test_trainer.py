from pso_nn_coupling.nn_trainer_with_pso import NNTrainerUsingPSO
from pso.pso import PSOConfig
from data_prep.data_prep import *
from nn.nn import NeuralNetwork
from nn.config_models import *
from pso.config_models import *
from data_prep.input_data_models import *
from pso.constants import *
from data_prep.constants import *
from nn.constants import *
import csv

data_config = DataPrepConfig(
    norm_method = NormMethod.ZSCORE,
    norm_factors = NORM_DEFAULT_FACTORS[NormMethod.ZSCORE]
)

nn_config = NNConfig(
    input_dim = 8,
    layers_sizes = [8, 1],
    activation_functions = [ActFunc.RELU, ActFunc.LINEAR],
    cost_function = CostFunc.MEAN_SQUARED_ERROR
)
pso_config = PSOConfig(
    max_iter = 500,
    swarm_size = 20,
    informant_count = 5,

    boundary_handling = BoundHandling.REFLECT,
    informant_selection = InformantSelect.STATIC_RANDOM,

    w_inertia = 0.5,

    c_personal = 1.3,
    c_social = 1.3,
    c_global = 1.3,

    jump_size = 1.0,
    dims = 8,                
    boundary_min = [],       
    boundary_max = [],       
    target_fitness = None,
)

def test_nn_trainer_using_pso_runs():
    data_prep = DataPrep(data_config)
    training_points, testing_points = data_prep.get_normalized_input_data_split()

    # Last column is target
    training_data_features = [p.features_norm_values for p in training_points]
    training_data_target = [p.target_norm_value for p in training_points]
    training_data = {'inputs': training_data_features, 'targets': training_data_target}

    # Last column is target
    testing_data_features = [p.features_norm_values for p in testing_points]
    testing_data_target = [p.target_norm_value for p in testing_points]
    testing_data = {'inputs': testing_data_features, 'targets': testing_data_target}

    pso_nn_trainer = NNTrainerUsingPSO(training_data, pso_config, nn_config)
    boundaries = pso_nn_trainer._calculate_pso_feature_boundaries()
    pso_config.dims = len(boundaries)
    pso_config.boundary_min = [b[0] for b in boundaries]
    pso_config.boundary_max = [b[1] for b in boundaries]
   
    best_position, best_training_cost = pso_nn_trainer.train_nn_using_pso()

    #_, bestf_pyswarm_pso = trainer.train_nn_using_pyswarm_pso()
    #_, bestf_pyswarm_pso_default = trainer.train_nn_pyswarm_pso_default()
    #print(f"Best Fitness of PYSWARM PSO : {bestf_pyswarm_pso}")
    #print(f"Best Fitness of PYSWARM PSO DEFAULT PARAMS: {bestf_pyswarm_pso_default}")

    nn = NeuralNetwork(config=nn_config)
    trained_nn_w, trained_nn_b = pso_nn_trainer._pso_vector_to_nn_weights_and_biases(np.array(best_position))
    test_cost = nn.get_cost_full_set(
        trained_nn_w, trained_nn_b, testing_data['inputs'], testing_data['targets'])
    print(f"---- Best training cost : {best_training_cost}")
    print(f"---- Test cost          : {test_cost}")
    generalization_ration = test_cost / best_training_cost
    print(f"---- Test cost / Best training cost: {generalization_ration}")
    if(generalization_ration > 1.5):
        print("WARNING: Possible overfitting detected (Test cost significantly higher than training cost)")
    elif(generalization_ration > 1.2):
        print("No overfitting detected (Test cost within acceptable range of training cost)")
    else:
        print("Good generalization (Test cost close to training cost)")

    ####################### Predictions on real values
    predictions = nn.get_predictions_whole_set(
        trained_nn_w, trained_nn_b, testing_data['inputs'])
    predictions_real_values = predictions * data_prep.normalization_factors[1][-1] + data_prep.normalization_factors[0][-1]
    mse_from_prediction = sum((pred - target_real) ** 2 for pred, target_real in zip(predictions_real_values, [p.target_real_value for p in testing_points])) / len(testing_points)
    print(f"---- MSE on real values : {mse_from_prediction}")

    mse_from_mean = sum((
        (data_prep.normalization_factors[0][-1] - target_real) ** 2 for 
        target_real in [p.target_real_value for p in testing_points])) / len(testing_points)

    print(f"---- coefficient of realization : {1 - (mse_from_prediction / mse_from_mean)}")

    write_targets_and_predictions(
        testing_points, 
        predictions_real_values, 
        TEST_OUTPUT_DIR + "predictions_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    )


def write_targets_and_predictions(points, predictions, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['target_real_value', 'prediction'])
        for point, pred in zip(points, predictions):
            writer.writerow([point.target_real_value, pred])

if __name__ == "__main__":
    import time
    t0 = time.time()
    test_nn_trainer_using_pso_runs()
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0}")
