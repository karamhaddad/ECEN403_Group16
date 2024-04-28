#include "cache.h"
#include "tensorflow/c/c_api.h"
#include <vector>
#include <bitset>
#include <memory>
#include <ostream>
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph.h"

#include <iostream>

using namespace tensorflow;

//GraphDef graph_def;

static std::vector<float> previous_input_vector;    // Buffer for the previous input vector

const int Page_number_size = 14;                    // Sets Page Number to 14 bits
const int Page_offset_size = 12;                    // Sets Page Offset to 12 bits
const int Cycle_delta_size = 22;                    // Sets Cycle Delta to 22 bits
const int Type_size = 1;                            // Type 1 bit

unsigned int Cycle = 0;                             // Global variable for Cycle
unsigned int Cycle_buf = 0;                         // Global variable for Previous Cycle

tensorflow::SessionOptions session_options_ = tensorflow::SessionOptions();                     // Configuration Options for TF session
tensorflow::RunOptions run_options_ = tensorflow::RunOptions();                                 // Run-time options for TF session (debug)
tensorflow::SavedModelBundle model_ = tensorflow::SavedModelBundle();                           // Object where saved model is stored
//tensorflow::Status status;
std::unique_ptr<Session> session;                                                               // Pointer for session (to be used)
const std::string export_dir = "/mnt/md0/jupyter/students/nathanielbush/test/model4/model";     // Directory for model


// Tensor building function
tensorflow::Tensor build_input_tensor(const std::vector<float>& prev_input, const std::vector<float>& curr_input) 
{
    //std::cout << "Building input tensor startup" << std::endl;
    // Create a tensor with the combined input setting shape to specified input for 1 sample, 2 steps, 29 features
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 2, 49}));
    // Retrieces mutable view of model, which specifies float data type adn dimension as 3
    auto input_tensor_map = input_tensor.tensor<float, 3>();

    // Copy the previous and the current input into the Tensor

    for (int i = 0; i < 49; ++i) {
        input_tensor_map(0, 0, i) = prev_input[i]; // Previous input at timestep 0
        input_tensor_map(0, 1, i) = curr_input[i]; // Current input at timestep 1
    }

    //std::cout << "Building input tensor end" << std::endl;

    return input_tensor;

}

// Function to convert an input to a binary vector
std::vector<float> int_to_binary_vector(uint64_t value, int bit_size) 
{
    //std::cout << "int to binary startup" << std::endl;
    // Vector initialized with 'bit_size' elements, all initially 0
    std::vector<float> binary_vector(bit_size, 0);
    // Value expression computes ith position of the int value, shifts to right
    for (int i = 0; i < bit_size; ++i) {
        binary_vector[bit_size - i - 1] = (value >> i) & 1;
    }
    //std::cout << "int to binary end" << std::endl;
    return binary_vector;
}

std::vector<float> process_input1(uint64_t addr, uint8_t type) 
{
    //std::cout << "process input startup" << std::endl;
    //std::cout << std::flush;

    // Extract the page number and the offset
    uint64_t page_number = addr >> 12;          // Right shift by 12 to get the page address
    uint64_t page_offset = addr & 0xFFF;        // Mask for the offset

    // Get the cycle delta
    uint64_t cycle_delta = Cycle - Cycle_buf;   // Uses global variables to find cycle delta
    Cycle_buf = Cycle;                          // Sets buffer to current

    // Convert to binary format using function
    std::vector<float> page_number_vector = int_to_binary_vector(page_number, Page_number_size);
    std::vector<float> page_offset_vector = int_to_binary_vector(page_offset, Page_offset_size);
    std::vector<float> cycle_delta_vector = int_to_binary_vector(cycle_delta, Cycle_delta_size);
    std::vector<float> type_vector = int_to_binary_vector(type, Type_size);

    // Combine all the parts to get a single binary vector for the input
    std::vector<float> input_vector;
    input_vector.reserve(Page_number_size + Page_offset_size + Cycle_delta_size + Type_size);

    // Convert the bitsets to vector <float>
    for(int i = Page_number_size - 1; i >= 0; --i) {
        input_vector.push_back(page_number_vector[i]);
    }
    for(int i = Page_offset_size - 1; i >= 0; --i) {
        input_vector.push_back(page_offset_vector[i]);
    }
    for(int i = Cycle_delta_size - 1; i >= 0; --i) {
        input_vector.push_back(cycle_delta_vector[i]);     
    } 
    input_vector.push_back(type_vector[0]);
    //std::cout << "process input end" << std::endl;

    return input_vector;

}

// Converts the binary vector Page Offset output into a whole address 
uint64_t process_output(uint64_t current_addr, const std::vector<float>& output_bits)
{
    
    // Initialize a result to 0
    uint64_t result = 0;
    // TODO: check what to initialize the i variable to, 1 or 0
    // Set each bit in output_bits to the correct position in result based on order
    for (size_t i = 1; i < output_bits.size(); ++i)
    {
        // adds shifted bit to result 
        result |= static_cast<uint64_t>(output_bits[i]) << (output_bits.size() - 1 - i);
    }
    // Extract the page_number from current address
    uint64_t page_number = current_addr >> 12;
    // Combine the page numer with result using OR
    return (page_number << 12) | result;
}

std::vector<float> tensor_to_vector(const Tensor& output_tensor) 
{
    // Convert the output tensor using flat() (accesses tensor data as flat array, disregarding original shape)
    auto flat = output_tensor.flat<float>();
    // flat.data() calls pointer, flat.size() finds size, start at beginning of pointer and go to end using the size
    return std::vector<float>(flat.data(), flat.data() + flat.size());
}

void CACHE::prefetcher_initialize() {
    //std::cout << "prefetcher initialize startup" << std::endl;

    // Attempts to load the model from the directory ({"serve"} sets to run only)
    auto status = tensorflow::LoadSavedModel(session_options_, 
                                                    run_options_, 
                                                    export_dir, 
                                                    {"serve"}, 
                                                    &model_);

    // Checks to see if the status is ok
    if (!status.ok()) {
        std::cerr << "Failed to load saved model: " << status.ToString() << std::endl;
        return; // Handle error appropriately
    }

    //std::cout << "end load" << std::endl;
    

    // Uncomment to find the node names in the model, this is to name the input and output node names correctly
    //  // Access the graph directly from the model
    // GraphDef graph_def = model_.meta_graph_def.graph_def();

    // // Print all node names in the graph
    // for (const NodeDef& node : graph_def.node()) {
    //     std::cout << "Node name: " << node.name() << std::endl;
    // }


    // The following code snippet was used to solve the segmentation fault problem caused by initializing the NN in 
    // one function, and running it in another.

    // Create fake variables
    // uint64_t addr1 = 1076800;
    // uint64_t addr2 = 1076800;
    // uint8_t type1 = 0;
    // unsigned int cycle1 = 0;
    // unsigned int cycle2 = 0;

    // Turn fake variables into vector
    // std::vector<float> previous_input_vector2 = process_input1(addr1, type1);
    // std::vector<float> current_input_vector2 = process_input1(addr2, type1);

    // Check if prevous was empty, shouldn't run 
    // if(previous_input_vector.empty()) {
    //     previous_input_vector.resize(Page_number_size + Page_offset_size + Cycle_delta_size + Type_size);
    // }

    // // get signatures
    // if (!status.ok()) {
    //     std::cerr << "Failed: " << status;
    // }

    // // Create the tensor for the input
    // std::cout << "Fake tensor input check startup" << std::endl;
    // tensorflow::Tensor fake_input_tensor = build_input_tensor(previous_input_vector2, current_input_vector2);
    // std::vector<Tensor> fake_output_tensor;
    
    // std::cout << "Running model" << std::endl;
    
    // Create input vector
    // std::vector<std::pair<string, tensorflow::Tensor >> inputs = {{"serving_default_lstm_input", fake_input_tensor}};

    // // Run the tensorflow model
    // tensorflow::Status run_status;
    // run_status = model_.session->Run(inputs, {"StatefulPartitionedCall"}, {}, &fake_output_tensor);

    // if (!run_status.ok()) {
    //     std::cerr << "Inference failed: " << run_status;
    //     return;
    // }
    // // Assigning session from the loaded model
    // session.reset(model_.session.release());
    // std::cout << "end model creation" << std::endl;
    // std::cout << "Fake tensor input check finish" << std::endl;
}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in) 
{
    //std::cout << "cache_operate startup" << std::endl;

    // Prepare inputs for the tensor //

    // Creates an input vector from the addr and type using process_input1()
    std::vector<float> current_input_vector = process_input1(addr, type);

    // Checks if previous vector was empty, if so, resize to 3-D vector
    if(previous_input_vector.empty()) {
        previous_input_vector.resize(Page_number_size + Page_offset_size + Cycle_delta_size + Type_size);
    }

    // Create an input tensor from the previous and current input vectors, used to input into machine
    tensorflow::Tensor input_tensor = build_input_tensor(previous_input_vector, current_input_vector);

    // Update the previous input vector for the next run
    previous_input_vector = current_input_vector;

    // Model startup and load //
    
    // Attemps to load the pretrained mode with given parameters
    auto status = tensorflow::LoadSavedModel(session_options_, 
                                                    run_options_, 
                                                    export_dir, 
                                                    {"serve"}, 
                                                    &model_);

    // Resize the previous input vector again (idk why its here again but it is)
    if(previous_input_vector.empty()) {
        previous_input_vector.resize(Page_number_size + Page_offset_size + Cycle_delta_size + Type_size);
    }

    // Checks to see the model was loaded correctly
    if (!status.ok()) {
        std::cerr << "Failed to load saved model: " << status.ToString() << std::endl;
        return metadata_in; // Handle error appropriately
    }

    // Run the tensorflow model//

    // Prepares the input for the TF session using the name of the correct node
    std::vector<std::pair<string, tensorflow::Tensor >> inputs = {{"serving_default_lstm_input", input_tensor}};
    
    // Creates an output tensor for the output of the session
    std::vector<Tensor> outputs;
    // Runs the TF session with the correct names of the input and output nodes, and output tensor
    tensorflow::Status run_status = model_.session->Run(inputs, {"StatefulPartitionedCall"}, {}, &outputs);
    // Checks if the model was correctly ran
    if (!run_status.ok()) {
        std::cerr << "Failed to run TensorFlow session: " << run_status.ToString() << "\n";
        return metadata_in;
    } 

    // Unused variable for output tensor for debugging purposes (extracts for tensor from list)
    auto output_tensor = outputs[0].tensor<float, 2>();
    
    // Convert the output tensor to a vector<float> for easier manipulation
    std::vector<float> output_vector = tensor_to_vector(outputs[0]);

    // Convert floating-point probabilities to binary (thresholding at 0.5) 
    std::vector<float> binary_output(output_vector.size());
    std::transform(output_vector.begin(), output_vector.end(), binary_output.begin(), [](float value) {
        return value > 0.5 ? 1 : 0;
    });


    // Used to print what the output of the NN was for debug purposes
    std::cout << "Predicted output vector: ";
    for (float value : binary_output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Process the output to find the complete prefetch address
    uint64_t result = process_output(addr, binary_output);

    // Send it off to the big wide world of the L2 Cache
    prefetch_line(result, true, metadata_in);

    // Assigning session from the loaded model
    session.reset(model_.session.release());
    
    //TODO: Set up the session close section once the program is done running
    //std::cout << "cache_operate fin" << std::endl;
    return metadata_in;

}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {
    Cycle++;        // Updates for cycle time and deltas
}

void CACHE::prefetcher_final_stats() {}


