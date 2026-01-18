import numpy as np
import os

def export_to_header(x_path, y_path, index, header_name="include/sample_data.h"):
    # Load the datasets
    try:
        x_test = np.load(x_path)
        # Assuming Y_test exists, otherwise handle gracefully
        y_test = np.load(y_path) if os.path.exists(y_path) else None
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Check index bounds
    if index >= len(x_test):
        print(f"Index {index} is out of bounds for dataset size {len(x_test)}")
        return

    # Select the sample
    sample_x = x_test[index]
    label = y_test[index] if y_test is not None else "N/A"

    # Flatten the sample if it's multidimensional (13x64 -> 832 elements)
    flattened_data = sample_x.flatten()
    
    with open(header_name, 'w') as f:
        f.write(f"/* Auto-generated header for sample index {index} */\n")
        f.write(f"/* Label: {label} */\n\n")
        f.write(f"#ifndef SAMPLE_DATA_H\n")
        f.write(f"#define SAMPLE_DATA_H\n\n")
        
        # Define the array size
        f.write(f"#define SAMPLE_SIZE {flattened_data.size}\n\n")
        
        # Write the C array
        f.write(f"float sample_input[SAMPLE_SIZE] = {{\n    ")
        
        for i, value in enumerate(flattened_data):
            f.write(f"{value:f}f")
            if i < flattened_data.size - 1:
                f.write(", ")
            # Add line breaks every 8 values for readability
            if (i + 1) % 8 == 0:
                f.write("\n    ")
        
        f.write("\n};\n\n")
        f.write(f"#endif // SAMPLE_DATA_H\n")

    print(f"Successfully exported sample {index} to {header_name}")

# Usage
export_to_header('DSP/DataSets/X_test.npy', 'DSP/DataSets/y_test.npy', index=0)