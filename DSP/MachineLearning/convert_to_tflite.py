#!/usr/bin/env python3
"""
Convert Keras model to TensorFlow Lite for ESP32 deployment
"""

import tensorflow as tf
import os
import json

def convert_model_to_tflite():
    """Convert the best Keras model to TFLite format"""
    
    # Path to the best model
    model_path = "model_weights/best_models/voiceModel_FullTrain_3classes_SMALL.keras"
    output_dir = "model_weights/tflite_models"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Print model summary
    print("\n=== Model Summary ===")
    model.summary()
    
    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure converter for better ESP32 compatibility
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert to quantized model (reduces size significantly)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_output_path = os.path.join(output_dir, "voiceModel_3classes_small.tflite")
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved to: {tflite_output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    # Print normalization parameters for C++ usage
    norm_path = "model_weights/norm_params.json"
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    print("\n=== Normalization Parameters (for C++) ===")
    print(f"#define MFCC_MEAN {norm_params['mean']:.10f}f")
    print(f"#define MFCC_STD {norm_params['std']:.10f}f")
    
    # Print model input/output shapes
    print("\n=== Model Input/Output ===")
    print(f"Input shape: (1, 13, 64, 1)")  # (batch, mfcc_coeffs, time_steps, channels)
    print(f"Output shape: (1, 3)")  # (batch, num_classes)
    print(f"Number of classes: 3")
    print(f"Classes: ambiente, apagarLuz, prenderLuz")

if __name__ == "__main__":
    convert_model_to_tflite()
