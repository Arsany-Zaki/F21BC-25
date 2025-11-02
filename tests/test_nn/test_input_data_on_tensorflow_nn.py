import tensorflow as tf
from data_prep.data_prep import DataPrep

# Example config (you should load or define this as needed for your project)
CONFIG = {
    "data": {
        "input_data_file_path": "data/raw_input/concrete_data.csv",
        "columns": ["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age", "Strength"],
        "normalisation_method": "minmax",
        "normalisation_minmax_params": [0, 1],
        "split_test_size": 0.2
    },
    "output": {
        "log_level": "INFO"
    }
}

# Prepare data
preparator = DataPrep(CONFIG)
train_df, test_df = preparator.get_normalized_input_data_split()

# Assume last column is the target
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Build a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Print the cost at the last epoch
final_loss = history.history['loss'][-1]
print(f"Final training loss (MSE) at last epoch: {final_loss:.6f}")