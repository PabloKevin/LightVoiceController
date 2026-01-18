#!/usr/bin/env python3
"""
Convert Keras model to TensorFlow Lite for ESP32 deployment
"""
import tensorflow as tf
import numpy as np

def convert_model_to_tflite():
    model = tf.keras.models.load_model("model_weights/best_models/voiceModel_FullTrain_3classes_SMALL.keras")
    x_test = np.load("/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/X_train.npy").astype(np.float32)

    def representative_data_gen():
        for i in range(min(100, len(x_test))):
            # Pick one sample and ensure it is (1, 13, 64, 1)
            sample = x_test[i].reshape(1, 13, 64, 1)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Force full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open("model_weights/tflite_models/voiceModel_3classes_small.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Success! Model converted with 4D calibration data.")


if __name__ == "__main__":
    convert_model_to_tflite()
