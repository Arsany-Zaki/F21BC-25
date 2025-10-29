from __future__ import annotations
from typing import List, Tuple
import logging
import pandas as pd
import numpy as np

class DataPreparator:
    def __init__(self, config: dict):
        self.config = config
        log_level = self.config["output"]["log_level"]
        logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))

    def _read_input_data(self) -> pd.DataFrame:
        data_file_path: str = self.config["data"]["input_data_file_path"]
        expected_columns: List[str] = self.config["data"]["columns"]
        raw_data = pd.read_csv(data_file_path)
        raw_data.columns = expected_columns
        return raw_data

    def _normalize_input_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        normalization_method = str(self.config["data"]["normalisation_method"]).lower().strip()
        logging.getLogger(__name__).info("Normalizing all columns")
        raw_data = raw_data.astype(float)
        normalized_data = raw_data.copy()

        if normalization_method == "zscore":
            target_mean, target_std = self.config["data"]["normalisation_zscore_params"]
            target_mean = float(target_mean)
            target_std = float(target_std)
            means = raw_data.mean()
            stds = raw_data.std()
            zero_var = stds == 0.0 # to avoid division by zero
            safe_stds = stds.copy()
            safe_stds[zero_var] = 1.0 # arbitrary non-zero value
            normalized_data.loc[:, raw_data.columns] = ((raw_data - means) / safe_stds) * target_std + target_mean
        elif normalization_method == "minmax":
            min_value, max_value = self.config["data"]["normalisation_minmax_params"]
            min_value = float(min_value)
            max_value = float(max_value)
            mins = raw_data.min()
            maxs = raw_data.max()
            ranges = maxs - mins
            zero_range = ranges == 0.0 # to avoid division by zero
            safe_ranges = ranges.copy()
            safe_ranges[zero_range] = 1.0 # arbitrary non-zero value
            scaled = (raw_data - mins) / safe_ranges
            normalized_data.loc[:, raw_data.columns] = scaled * (max_value - min_value) + min_value
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")
        return normalized_data

    def _split_data_to_training_testing(self, normalized_data: pd.DataFrame, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_fraction = float(self.config["data"]["split_test_size"])
        total_records = len(normalized_data)
        random_generator = np.random.RandomState(seed)
        indices = np.arange(total_records)
        random_generator.shuffle(indices)

        num_test = int(round(total_records * split_fraction))
        test_indices = indices[:num_test]
        train_indices = indices[num_test:]

        train_data = normalized_data.iloc[train_indices].reset_index(drop=True)
        test_data = normalized_data.iloc[test_indices].reset_index(drop=True)
        return train_data, test_data

    def get_raw_input_data(self) -> pd.DataFrame:
        if not hasattr(self, '_raw_input_data'):
            raise ValueError("Raw input data not available. Call prepare_data_for_training() first.")
        return self._raw_input_data
    
    def get_normalized_input_data(self) -> pd.DataFrame:
        if not hasattr(self, '_normalized_input_data'):
            raise ValueError("Normalized input data not available. Call prepare_data_for_training() first.")
        return self._normalized_input_data
    
    def get_normalized_input_data_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, '_training_data') or not hasattr(self, '_testing_data'):
            raise ValueError("Training/testing data not available. Call prepare_data_for_training() first.")
        return self._training_data, self._testing_data

    def prepare_data_for_training(self):
        self._raw_input_data = self._read_input_data()
        self._normalized_input_data = self._normalize_input_data(self._raw_input_data)
        self._training_data, self._testing_data = self._split_data_to_training_testing(self._normalized_input_data)
        