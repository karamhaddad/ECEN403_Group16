import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split


# Global declaration of the outputs list
#outputs = []
loss_iteration = 0

# Custom Loss Function
def custom_loss(y_true, y_pred):

    print(y_true)
    print(y_pred)

    loss_iteration += 1
    return standard_loss + 0.1 * custom_component

# Input Preprocessing
def input_process(line):

    parts = line.strip().split()                    # Splits the current line into parts and removes any space

    if len(parts) < 4:                              # Safety check for incorrect line length, should be 4 parts
        return None
    
    input_sequence = ''.join([                      # Create one long string by joining the parts, zfill() used to make sure each part is the correct length
        parts[0].zfill(14),                         # Page Address should be 14 bits
        parts[1].zfill(12),                         # Page Offset should be 12 bits
        parts[2].zfill(22),                         # Cycle Delta should be 22 bits
        parts[3].zfill(1)                           # Type will always be 1 bit
    ])
    return [int(bit) for bit in input_sequence]     # Iterates over each character in string, returns a binary int

# Output Preprocessing
def output_process(line):
    parts = line.strip().split()                    # Splits the current line into parts
    output_sequence = parts[1].zfill(12)            # Page Offset should be 12 bits
    return [int(bit) for bit in output_sequence]    # Iterates over each character in string, returns a binary int

# File Preprocessing
def preprocess_file(file_path):
    #global outputs                                  # Declares the global variable to be used in the function
    #outputs.clear()                                 # Clears global outputs list 

    with open(file_path, 'r') as file:              # Opens the file at the file_path
        lines = file.readlines()                    # Reads all lines from the list, each element is a string 

    sequences = []                                  # Initializes a list to store the input sequences
    outputs = []                                    # Initializes a list to store the output sequences

    for i in range(1, len(lines) - 1):              # Start reading at second line to avoid segmentation faults

        prev_input = input_process(lines[i-1])      # Process the previous line
        curr_input = input_process(lines[i])        # Process the current line

        if prev_input is not None and curr_input is not None:   # Ensures only valid inputs are permitted into the LSTM Model training

            input_sequence = prev_input + curr_input            # Concatenates the previous and current process lines to be one input
            output_sequence = output_process(lines[i+1])        # Calls the output_process function for the next line
            sequences.append(input_sequence)                    # Appends the input
            outputs.append(output_sequence)                     # Appends the output

    return np.array(sequences), np.array(outputs)               # Returns the two arrays

def build_model(hp):
    model = Sequential([
        LSTM(
            units=hp.Int('first_lstm_units', min_value=180, max_value=220, step=10),    # First layer to range around 192 for finer tuning
            input_shape=(2, 49),
            return_sequences=True
        ),
        LSTM(
            units=hp.Int('second_lstm_units', min_value=180, max_value=220, step=10),   # Second layer to range around 192 for finer tuning
            return_sequences=True
        ),
        LSTM(
            units=hp.Int('third_lstm_units', min_value=50, max_value=100, step=10),     # Third layer to range around 62 for finer tuning
            return_sequences=False
        ),
        Dense(12, activation='sigmoid')                                                 # Last Dense (fully connected) layer, Sigmoid converts values to be from 0 - 1
    ])
    
    # Configure the Adam optimizer with a narrower focus on learning rate
    adam_optimizer = Adam(
        learning_rate=hp.Float('learning_rate', min_value=0.0001, max_value=0.001, sampling='log'),     # Finer tuning around a specific range
        beta_1=hp.Float('beta_1', min_value=0.89, max_value=0.99),                                      # Narrow range if previous best was near default
        beta_2=hp.Float('beta_2', min_value=0.995, max_value=0.999)                                     # Narrow range if previous best was near default
    )
    
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])           # Compile with the same factors as when training
    return model

def tuner_setup():                              # This function sets up the tuner to run
    tuner = BayesianOptimization(               # Creates a class of 'BayesianOptimization'(Finds minimum of a function efficient enough)
        build_model,                            # Constructs a Keras model with a set of hyperparameters, object 'hp'
        objective='val_accuracy',               # Specifies what tuner should aim to optimize (validation-accuracy)
        max_trials=3,                           # Number of trials tested
        executions_per_trial=2,                 # More executions per trial for stability (times hyperparameters will be used)
        directory='finer_tuning',               # Specifies the directory of logs
        project_name='adam_fine_tuning'         # Project name
    )

    return tuner

if __name__ == "__main__":
    file_path = 'postprocess.txt'                              # Sets file path
    sequences, outputs = preprocess_file(file_path)             # Creates the input and output sequence arrays
    sequences = sequences.reshape((sequences.shape[0], 2, 49))  # Reshapes the input array to be 3-D

    # Splits data into training and testing sets, 80% training, 20% validating, random state ensures reproducability
    X_train, X_test, y_train, y_test = train_test_split(sequences, np.array(outputs), test_size=0.2, random_state=123)

    tuner = tuner_setup()                                       # Sets up the tuner object to be run for given hyperparameters

    # Starts the tuning process
    tuner.search(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

    
    best_model = tuner.get_best_models(num_models=1)[0]         # Get the best model from the trials

   
    loss, accuracy = best_model.evaluate(X_test, y_test)        # Evaluate the best model
    print(f"Best model loss: {loss}, accuracy: {accuracy}")

    # Save the best model
    best_model.save('best_adam_tuned_model.keras')
