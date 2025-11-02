import sys
import os
from utilities.path_manager import normalize_path, ensure_directory_exists, join_paths
from input_data_Processor.input_data_processor import DataPreparator
from datetime import datetime
from configs.paths import CONFIG

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_data_preparation():
    """Test the data preparation pipeline"""
    print("************* Data Preparation Test starts ************")
    # Merge configs as expected by DataPreparator
    data_preparator = DataPreparator(CONFIG.data)
    raw_input_data = data_preparator.get_raw_input_data()
    normalized_input_data = data_preparator.get_normalized_input_data()
    training_data, testing_data = data_preparator.get_normalized_input_data_split()
    
    # Get current timestamp in compact form
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data with timestamp in filenames
    output_dir = normalize_path(CONFIG.test.output_dir)
    ensure_directory_exists(output_dir)
    
    # Save raw data
    path = join_paths(output_dir, f"raw_data_{timestamp}.csv")
    raw_input_data.to_csv(path, index=False)
    print(f"{path} is saved")
    
    # Save normalized data
    path = join_paths(output_dir, f"normalized_input_data_{timestamp}.csv")
    normalized_input_data.to_csv(path, index=False)
    print(f"{path} is saved")
    
    # Save training data
    path = join_paths(output_dir, f"training_data_{timestamp}.csv")
    training_data.to_csv(path, index=False)
    print(f"{path} is saved")
    
    # Save testing data
    path = join_paths(output_dir, f"testing_data_{timestamp}.csv")
    testing_data.to_csv(path, index=False)
    print(f"{path} is saved")
    
    print("************* Data Preparation Test completed ************")

if __name__ == "__main__":
    test_data_preparation()