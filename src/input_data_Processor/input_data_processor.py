from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np
from configs.metadata import input_data_columns
from configs.paths import CONFIG
from configs.paths import DataConfig
from configs.metadata import NormMethod

class DataPreparator:
    def __init__(self, config: DataConfig):
        self.is_data_prepared: bool = False
        self.config: DataConfig = config

    def _read_input_data(self) -> pd.DataFrame:
        data_file_path: str = CONFIG.data.raw_input_dir + CONFIG.data.raw_input_file
        expected_columns: List[str] = list(input_data_columns.values())
        raw_data = pd.read_csv(data_file_path)
        raw_data.columns = expected_columns
        return raw_data

    def _normalize_input_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raw_data = raw_data.astype(float)
        normalized_data = raw_data.copy()

        if self.config.norm_method == NormMethod.ZSCORE:
            target_mean, target_std = self.config.norm_factors
            target_mean = float(target_mean)
            target_std = float(target_std)
            means = raw_data.mean()
            stds = raw_data.std()
            zero_var = stds == 0.0 # to avoid division by zero
            safe_stds = stds.copy()
            safe_stds[zero_var] = 1.0 # arbitrary non-zero value
            normalized_data.loc[:, raw_data.columns] = ((raw_data - means) / safe_stds) * target_std + target_mean
        elif self.config.norm_method == NormMethod.MINMAX:
            min_value, max_value = self.config.norm_factors
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
            raise ValueError(f"Unknown normalization method: {self.config.norm_method}")
        return normalized_data

    def _split_data_to_training_testing(self, normalized_data: pd.DataFrame, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_fraction = float(self.config.split_test_size)
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
        if not self.is_data_prepared:
            self._prepare_data_for_training()
        return self._raw_input_data

    
    def get_normalized_input_data(self) -> pd.DataFrame:
        if not self.is_data_prepared:
            self._prepare_data_for_training()
        return self._normalized_input_data

    def get_normalized_input_data_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.is_data_prepared:
            self._prepare_data_for_training()
        return self._training_data, self._testing_data

    def _prepare_data_for_training(self):
        if(self.is_data_prepared): # for safety
            return
        self._raw_input_data = self._read_input_data()
        self._normalized_input_data = self._normalize_input_data(self._raw_input_data)
        self._training_data, self._testing_data = self._split_data_to_training_testing(self._normalized_input_data)
        self.is_data_prepared = True
