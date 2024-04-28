
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        previous_cycle = None                       # To store the last number of the previous line
        
        for line in input_file:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[0] == "PREFETCH_TRAIN":

                # Extract numbers from files
                address = int(parts[1])             # Address consisting of Page Number and Page Offset
                type = int(parts[-2])               # Type which is either a read (0) or a write (1)
                current_cycle = int(parts[-1])      # Cycle time currently
                
                # Calculate page offset and page number
                page_offset = address % 4096        # Extract the Page Offset
                page_number = address >> 12         # Extract the Page Number
                
                # Calculate cycle time delta if possible 
                cycle_time_delta = current_cycle - previous_cycle if previous_cycle is not None else 0
                previous_cycle = current_cycle      # Update for next iteration
                
                # Convert to binary
                # bin() converts to binary starting with 0b, [2:] slices the first two characters
                page_offset_binary = bin(page_offset)[2:]
                page_number_binary = bin(page_number)[2:]
                cycle_time_delta_binary = bin(cycle_time_delta)[2:] if cycle_time_delta >= 0 else '-' + bin(-cycle_time_delta)[3:]  # Handle negative
                type_binary = bin(type)[2:]
                
                # Write to output file
                output_file.write(f"{page_number_binary} {page_offset_binary} {cycle_time_delta_binary} {type_binary}\n")

# Example usage
input_file_path = '/Users/nathanielbush/Desktop/ECEN403/NeuralNetwork/Data/600.perlbench_s-1273B.champsimtrace.xz_none_cli_0_l2clog.txt'
output_file_path = 'postprocess3.txt'
process_file(input_file_path, output_file_path)
