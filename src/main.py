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

# Input data normalization configuration
data_config = DataPrepConfig(
    norm_method = NormMethod.ZSCORE,
    norm_factors = NORM_DEFAULT_FACTORS[NormMethod.ZSCORE]
)
# Neural Network configuration
nn_config = NNConfig(
    input_dim = 8,
    layers_sizes = [8, 1],
    activation_functions = [ActFunc.RELU, ActFunc.LINEAR],
    cost_function = CostFunc.MEAN_SQUARED_ERROR
)
# PSO configuration
pso_config = PSOConfig(
    max_iter = 20,
    swarm_size = 10,
    informant_count = 2,

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

def main():
    # Read data from file and prepare it (normalize and split into training and testing sets)
    print("Get normalized input data split...", end="\n\n")
    data_prep = DataPrep(data_config)
    training_points, testing_points = data_prep.get_normalized_input_data_split()

    # Train NN using PSO and training data set
    print("Train neural network using PSO...", end="\n\n")
    _print_all_configs()
    pso_nn_trainer = NNTrainerUsingPSO(training_points, pso_config, nn_config)   
    best_position, best_training_cost = pso_nn_trainer.train_nn_using_pso()

    # Evaluate neural network on test set (normalized values)
    print("Evaluate neural network on test set...", end="\n\n")
    nn = NeuralNetwork(config=nn_config)
    trained_nn_weights, trained_nn_biases = pso_nn_trainer.pso_vector_to_nn_weights_and_biases(np.array(best_position))
    test_cost, test_predictions = nn.forward_run_full_set(trained_nn_weights, trained_nn_biases, testing_points)
    generalization_ratio = test_cost / best_training_cost
    _print_evaluation_metrics_on_norm_data(best_training_cost, test_cost)

    # Convert neural network predictions to real (non-normalized) values
    norm_factor_mean = data_prep.normalization_factors[0][-1]
    norm_factor_std = data_prep.normalization_factors[1][-1]
    nn_predictions_real_vals = test_predictions * norm_factor_std + norm_factor_mean
    test_target = [p.target_real_value for p in testing_points]
    
    # Calculate neural network evaluation metrics on real (non-normalized) values
    se_nn = [(pred - target_real) ** 2 for pred, target_real in zip(nn_predictions_real_vals, test_target)]
    mse_nn = sum(se_nn) / len(testing_points)
    rmse_nn = mse_nn ** 0.5
    mae_nn = sum(abs(pred - target_real) for pred, target_real in zip(nn_predictions_real_vals, test_target)) / len(testing_points)
    msg = "Neural network evaluation metrics on real (non-normalized) values"
    _print_standard_evaluation_metrics(msg, mse_nn, rmse_nn, mae_nn)

    # Calculate baseline model (mean of training targets) evaluation metrics on real (non-normalized) values
    training_target_mean = sum(p.target_real_value for p in training_points) / len(training_points)
    se_baseline = [(training_target_mean - target_real) ** 2 for target_real in test_target]
    mse_baseline = sum(se_baseline) / len(testing_points)
    rmse_baseline = mse_baseline ** 0.5
    mae_baseline = sum(abs(training_target_mean - target_real) for target_real in test_target) / len(testing_points)
    msg = "Baseline model (mean) evaluation metrics on real (non-normalized) values"
    _print_standard_evaluation_metrics(msg, mse_baseline, rmse_baseline, mae_baseline)

    # Calculate relative performance metrics
    coefficient_of_realization = 1 - (mse_nn / mse_baseline)
    _print_comparison_metrics(coefficient_of_realization, generalization_ratio)

    # Write predictions to CSV file
    write_targets_and_predictions(
        testing_points, 
        nn_predictions_real_vals, 
        TEST_OUTPUT_DIR + "predictions_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"
    )

def _print_standard_evaluation_metrics(msg, mse, rmse, mae):
    print(f"***** {msg}")
    print(f"- MSE  : {mse}")
    print(f"- RMSE : {rmse}")
    print(f"- MAE  : {mae}")
    print()

def _print_comparison_metrics(coefficient_of_realization, generalization_ratio):
    print("***** Model Performance Comparison")
    print(f"- Coefficient of Realization : {coefficient_of_realization:.4f}")
    print(f"- Generalization Ratio       : {generalization_ratio:.4f}")
    print()

def _print_all_configs():
    print("***** Configuration Settings")
    print("- Data Preparation Config:")
    print(data_config)
    print()
    print("- Neural Network Config:")
    print(nn_config)
    print()
    print("- PSO Config:")
    print(pso_config)
    print()

def _print_evaluation_metrics_on_norm_data(best_training_cost, test_cost):
    print("***** Evaluation Metrics on Normalized Data")
    print(f"- Cost function is   : {nn_config.cost_function}")
    print(f"- Best training cost : {best_training_cost}")
    print(f"- Test cost          : {test_cost}")
    print()

def write_targets_and_predictions(points, predictions, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['target_real_value', 'prediction'])
        for point, pred in zip(points, predictions):
            writer.writerow([point.target_real_value, pred])

if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0}")
