import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

# Global declaration of the outputs list
outputs = []

# Custom Loss Function (currently none functioning)
def custom_loss(y_true, y_pred):
    standard_loss = BinaryCrossentropy()(y_true, y_pred)
    custom_component = tf.reduce_mean(tf.square(y_true - y_pred))
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
    global outputs                                  # Declares the global variable to be used in the function
    outputs.clear()                                 # Clears global outputs list 

    with open(file_path, 'r') as file:              # Opens the file at the file_path
        lines = file.readlines()                    # Reads all lines from the list, each element is a string 

    sequences = []                                  # Initializes a list to be store the input sequences

    for i in range(1, len(lines) - 1):              # Start reading at second line to avoid segmentation faults

        prev_input = input_process(lines[i-1])      # Process the previous line
        curr_input = input_process(lines[i])        # Process the current line

        if prev_input is not None and curr_input is not None:   # Ensures only valid inputs are permitted into the LSTM Model training

            input_sequence = prev_input + curr_input            # Concatenates the previous and current process lines to be one input
            output_sequence = output_process(lines[i+1])        # Calls the output_process function for the next line
            sequences.append(input_sequence)                    # Appends the input
            outputs.append(output_sequence)                     # Appends the output

    return np.array(sequences), np.array(outputs)               # Returns the two arrays

# LSTM Model Definition using best hyperparameters
def create_lstm_model():
    model = Sequential([                                        # Initialize a new 'Sequential' model (linear stack of layers in Keras)   
        LSTM(192, input_shape=(2, 49), return_sequences=True),  # Add LSTM layer of 192 units, input shape() specifies input data shape, 2 time steps, 49 features, r_s means the full sequence is returned in the next layer
        LSTM(192, return_sequences=True),                       # Second LSTM layer which returns sequences, will pass only to next layer rather than output
        LSTM(64, return_sequences=False),                       # Third LSTM layer, only returns the output of the last time step, into a single vector (transition layer)
        Dense(12, activation='sigmoid')                         # Last Dense (fully connected) layer, Sigmoid converts values to be from 0 - 1
    ])

    # Using best learning rate found from tuning (example: 0.0005)
    adam_optimizer = Adam(learning_rate=0.0005)                 # Uses ADAM optimizer (good for binary), learning rate controls update of model in response to estimation error when model weights updated
    model.compile(optimizer=adam_optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])    # Configures for training, ADAM, BCE loss function (good for binary), accuracy shows when prediction equal labels
    return model

# Main script
if __name__ == "__main__":
    file_path = 'postprocess2.txt'                               # Input file path   
    sequences, outputs = preprocess_file(file_path)             # Creates input and output data arrays from the preprocessing function
    sequences = sequences.reshape((sequences.shape[0], 2, 49))  # Reshape the sequence array to a 3-D shape suitable for LSTM model (number of samples, time steps, features)

    # Splits data into training and testing sets, 80% training, 20% validating, random state ensures reproducability
    X_train, X_test, y_train, y_test = train_test_split(sequences, outputs, test_size=0.2, random_state=123)
    model = create_lstm_model()                                 # Calls function used to create the model
    
    # Trains the model using the specified data above, epochs = one complete data representation, batch is an iteration of the data
    model.fit(X_train, y_train, epochs=3, batch_size=1000, validation_data=(X_test, y_test))  # Adjust epochs and batch size as needed

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)             # Evalues the traied model on the test set, final validation 
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")      # print final loss and acuracy of the test data to the console

    # Save the trained model
    model.save('model6/model')              # Saves model for C++ use
    model.save('test6.keras')               # Saves model for Python use
