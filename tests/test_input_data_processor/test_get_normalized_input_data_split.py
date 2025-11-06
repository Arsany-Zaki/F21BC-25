from data_prep.constants import NormMethod
from data_prep.data_prep import DataPrep
from datetime import datetime
from config.paths import *
from data_prep.input_data_models import DataPrepConfig
import csv

def test_get_normalized_input_data_split():
    data_config = DataPrepConfig(
        norm_method = NormMethod.ZSCORE,
        norm_factors = [0, 1],
        split_test_size = 0.3,
        random_seed = 42
    )
    data_prep = DataPrep(data_config)
    training_data, testing_data = data_prep.get_normalized_input_data_split()   
    training_file_name = "normalized_training_data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    testing_file_name = "normalized_testing_data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    _write_points_to_csv(training_data, TEST_OUTPUT_DIR + training_file_name)
    _write_points_to_csv(testing_data, TEST_OUTPUT_DIR + testing_file_name)

def _write_points_to_csv(points, filename):
    num_features = len(points[0].features_real_values)
    num_norm_features = len(points[0].features_norm_values)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = (
            [f'feature_{i+1}' for i in range(num_features)] +
            ['target_real_value'] +
            [f'norm_feature_{i+1}' for i in range(num_norm_features)] +
            ['target_norm_value']
        )
        writer.writerow(header)
        for point in points:
            row = (
                list(point.features_real_values) +
                [point.target_real_value] +
                list(point.features_norm_values) +
                [point.target_norm_value]
            )
            writer.writerow(row)
if __name__ == "__main__":
    test_get_normalized_input_data_split()
    