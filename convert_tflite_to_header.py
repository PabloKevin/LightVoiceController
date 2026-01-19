#!/usr/bin/env python3
"""
Convert TFLite model to C header for embedding in ESP32
"""

import os
import sys

def tflite_to_c_array(tflite_path, output_header_path):
    """Convert .tflite file to C array"""
    
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    # Create C header
    with open(output_header_path, 'w') as f:
        f.write('#ifndef MODEL_DATA_H\n')
        f.write('#define MODEL_DATA_H\n\n')
        f.write('// TensorFlow Lite model for voice recognition\n')
        f.write('// Auto-generated from: {}\n'.format(tflite_path))
        f.write('// Size: {} bytes\n\n'.format(len(tflite_data)))
        
        # Write as aligned byte array
        f.write('const unsigned char voiceModel_3classes_tflite[] __attribute__((aligned(4))) = {\n')
        
        # Write bytes in rows of 16
        for i in range(0, len(tflite_data), 16):
            row = tflite_data[i:i+16]
            hex_bytes = ', '.join('0x{:02x}'.format(b) for b in row)
            f.write('    ' + hex_bytes)
            if i + 16 < len(tflite_data):
                f.write(',')
            f.write('\n')
        
        f.write('};\n\n')
        f.write('const unsigned int voiceModel_3classes_tflite_len = {};\n\n'.format(len(tflite_data)))
        f.write('#endif // MODEL_DATA_H\n')
    
    print(f"✓ C header file created: {output_header_path}")
    print(f"  Model size: {len(tflite_data) / 1024:.2f} KB")

if __name__ == "__main__":
    tflite_path = "DSP/MachineLearning/model_weights/tflite_models/voiceModel_3classes_small02.tflite"
    output_path = "include/model_data.h"
    
    if not os.path.exists(tflite_path):
        print(f"Error: TFLite model not found at {tflite_path}")
        sys.exit(1)
    
    tflite_to_c_array(tflite_path, output_path)
