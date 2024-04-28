import tensorflow as tf
import numpy as np

def input_process(line):
    parts = line.strip().split()
    # Check if line has 4 parts
    if len(parts) < 4:
        return None

    # Fill each individual number to base length
    input_sequence = ''.join([
        parts[0].zfill(14),         # Address fill to 14 bits
        parts[1].zfill(12),         # Offset fill to 12 bits
        parts[2].zfill(22),         # Cycle Delta to 22 bits
        parts[3].zfill(1)           # Fill to 1 bit
    ])

    # Convert the binary strings to a list of integers
    input_sequence_int = [int(bit) for bit in input_sequence]

    return input_sequence_int

# Main script
if __name__ == "__main__":
    file_path = 'postprocess.txt'                                   # File path 
    #print(tf.__version__)                                          # Prints Tensorflow version (debugging purposes)
    new_model = tf.keras.models.load_model('test2.keras')           # Loads the trained LSTM model to be checked
    previous_input = None                                           # Creates a variable to be used for the previous input

    # Loop for printing each varibles shape and name
    for var in new_model.variables:
        print(var.name, var.shape)

    # Prints summary of the model including layers, outputs, and the number of trainable parameters (debugging purposes)
    print(new_model.summary())

    max_iterations = 10                                             # Set the maximum number of iterations
    current_iteration = 0                                           # Initialize current iteration counter 

    # Open the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            if current_iteration >= max_iterations:                 # Set loop to stop after set time
                print("Reached the maximum number of iterations.")
                break                                               # Stop processing if the max iteration count is reached

            current_input = input_process(line)                     # Process the current line for input into the model

            if current_input is not None and previous_input is not None:        # Checks if both current and previous input are valid                    
                # Combine previous and current inputs to form a sequence of two timesteps
                input_sequence = np.array([previous_input, current_input])      # Combine previous and current inputs to form a sequence of two timesteps
                input_sequence = input_sequence.reshape((1, 2, 49))             # Reshape to (batch_size, timesteps, features)

                
                output = new_model.predict(input_sequence)          # Model prediction
                
                binary_output = (output > 0.5).astype(int)          # Convert sigmoid output probabilities to binary output (0 or 1)
                
                print(f"Predicted output (binary): {binary_output.flatten()}")      # Prints predicted binary output

            else:
                print("Invalid input or insufficient parts in line.")

            
            previous_input = current_input                          # Update previous input
            current_iteration += 1                                  # Increment the iteration counter
