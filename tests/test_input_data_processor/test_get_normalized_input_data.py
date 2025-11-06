from data_prep.constants import NormMethod
from data_prep.data_prep import DataPrep
from datetime import datetime
from config.paths import *
from data_prep.input_data_models import DataPrepConfig
import csv
from typing import List

def test_get_normalized_input_data():
    data_config = DataPrepConfig(
        norm_method = NormMethod.ZSCORE,
        norm_factors = [0, 1],
        split_test_size = 0.3,
        random_seed = 42
    )
    data_prep = DataPrep(data_config)
    norm_data = data_prep.get_normalized_input_data()
    file_name = "normalized_data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    _write_normalized_data_to_csv(norm_data, TEST_OUTPUT_DIR + file_name)

def _write_normalized_data_to_csv(rows: List[float], filename):
    num_features = len(rows[0]) - 1  # Last column is target
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = [f'norm_feature_{i+1}' for i in range(num_features)] + ['target_norm_value']
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    test_get_normalized_input_data()
    