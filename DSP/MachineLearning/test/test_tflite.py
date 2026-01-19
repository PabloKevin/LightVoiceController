import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
TFLITE_MODEL_PATH = "../model_weights/tflite_models/voiceModel_3classes_small02.tflite"
X_PATH = "../../DataSets/X_test.npy"
Y_PATH = "../../DataSets/y_test.npy"
CLASS_MAP_PATH = "../class_map.json"

def test_tflite_model():
    # 1. Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Check if the model is quantized (int8)
    is_quantized = input_details['dtype'] == np.int8

    # 2. Load Data
    X_test = np.load(X_PATH).astype(np.float32)
    mean, std =-50.281225, 159.534246
    X_test = (X_test - mean) / (std + 1e-8)
    y_test_labels = np.load(Y_PATH)
    
    with open(CLASS_MAP_PATH, 'r') as f:
        mapping = json.load(f)
    class_dict = mapping["CLASS_MAP"]
    class_names = list(class_dict.keys())
    
    # Map labels to indices
    y_true = np.array([class_dict[label] for label in y_test_labels])
    y_pred = []

    print(f"Testing {'Quantized' if is_quantized else 'Float'} TFLite Model...")

    for i in range(len(X_test)):
        # Prepare input: (1, 13, 64, 1)
        sample = X_test[i].reshape(input_details['shape'])

        # 3. MANUALLY QUANTIZE (if model is int8)
        if is_quantized:
            scale, zero_point = input_details['quantization']
            sample = sample / scale + zero_point
            sample = np.clip(np.round(sample), -128, 127).astype(np.int8)

        # 4. Set tensor and Invoke
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()

        # 5. Get output and MANUALLY DEQUANTIZE
        output_data = interpreter.get_tensor(output_details['index'])
        
        if is_quantized:
            scale, zero_point = output_details['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        # Get predicted class
        y_pred.append(np.argmax(output_data))

    # 6. Evaluation
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y_true)
    print(f"\nTFLite Model Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('TFLite Model Performance (Simulated on PC)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    test_tflite_model()