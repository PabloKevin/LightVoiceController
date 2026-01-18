import serial
import numpy as np
import struct
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'  # Change to your port (e.g., 'COM3' or '/dev/ttyACM0')
BAUD = 115200
X_PATH = '../DSP/DataSets/X_train.npy'
Y_PATH = '../DSP/DataSets/y_test.npy'
CLASS_MAP_PATH = '../DSP/MachineLearning/class_map.json'

def label2index(y_test, class_map):
    """Converts string labels to numeric indices using the class_map."""
    return np.array([class_map[label] for label in y_test])

def run_serial_test():
    # 1. Load Data
    print("--- Hardware-in-the-loop Test Bench ---")
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Error: Dataset files not found. Check X_PATH and Y_PATH.")
        return

    X_test = np.load(X_PATH).astype(np.float32)  # Expected shape: (90, 13, 64)
    y_test_labels = np.load(Y_PATH)
    
    with open(CLASS_MAP_PATH, 'r') as f:
        mapping = json.load(f)
    
    class_dict = mapping["CLASS_MAP"] # e.g., {"light_on": 0, "light_off": 1, ...}
    class_names = list(class_dict.keys())
    y_true = label2index(y_test_labels, class_dict)

    # 2. Open Serial Port
    try:
        # Increase timeout to ensure we don't drop during inference
        ser = serial.Serial(PORT, BAUD, timeout=5)
        print(f"Connected to {PORT}. Waiting for ESP32 to stabilize...")
        time.sleep(3) # Wait for bootloader messages to clear
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    y_pred = []
    confidences = []

    print(f"Starting inference for {len(X_test)} samples...")

    for i in range(len(X_test)):
        sample = X_test[i].flatten() # 832 floats
        
        # Send Start Marker
        ser.write(b'S') 
        
        # Pack 832 floats into 3328 bytes
        byte_data = struct.pack(f'{len(sample)}f', *sample)
        ser.write(byte_data)
        
        # Wait for the specific RESULT line
        found_result = False
        while not found_result:
            try:
                # errors='replace' handles the 0x80 / non-utf8 boot bytes
                line = ser.readline().decode('utf-8', errors='replace').strip()
                
                if "RESULT:" in line:
                    # Format: RESULT:predicted_class,confidence
                    data = line.split("RESULT:")[1]
                    p_class, p_conf = data.split(",")
                    
                    y_pred.append(int(p_class))
                    confidences.append(float(p_conf))
                    
                    print(f"[{i+1}/{len(X_test)}] Real: {y_test_labels[i]} | Pred: {class_names[int(p_class)]} ({float(p_conf):.2f})")
                    found_result = True
            except Exception as e:
                print(f"Error parsing line: {e}")
                break

    ser.close()

    # 3. Final Evaluation
    if len(y_pred) == len(y_true):
        plot_results(y_true, y_pred, class_names)
    else:
        print("Error: Mismatch between number of predictions and true labels.")

def plot_results(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('ESP32 Hardware-in-the-Loop Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('esp32_confusion_matrix.png')
    plt.show()
    
    print("\n--- Final Performance Metrics ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    run_serial_test()