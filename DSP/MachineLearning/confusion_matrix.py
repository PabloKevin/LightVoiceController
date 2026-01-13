import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tensorflow.keras.models import load_model

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_idx = label2index(y_test)
    
    # Cargar el archivo JSON
    with open('class_map.json', 'r') as file:
        mapping = json.load(file)

    # Acceder a los diccionarios
    classes = mapping["CLASS_MAP"].keys()

    cm = confusion_matrix(y_idx, y_pred_classes, len(classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.show()

def confusion_matrix(y_true, y_pred, num_classes):
    # Crear una matriz de ceros de tamaño (num_classes, num_classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Llenar la matriz
    for true_val, pred_val in zip(y_true, y_pred):
        cm[true_val, pred_val] += 1
        
    return cm

def index2label(indexes):
    # 2. Codificar etiquetas
    # Cargar el archivo JSON
    with open('class_map.json', 'r') as file:
        mapping = json.load(file)

    # Acceder a los diccionarios
    INDEX_TO_CLASS = mapping["INDEX_TO_CLASS"]

    # 2. Mapeo de etiquetas a indices
    labels = np.array([INDEX_TO_CLASS[label] for label in indexes])
    return labels

def label2index(labels):
    # 2. Codificar etiquetas
    # Cargar el archivo JSON
    with open('class_map.json', 'r') as file:
        mapping = json.load(file)

    # Acceder a los diccionarios
    CLASS_MAP = mapping["CLASS_MAP"]

    # 2. Mapeo de etiquetas a indices
    indexes = np.array([CLASS_MAP[label] for label in labels])
    return indexes

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "DataSets/")
    X = np.load(path + "X_test.npy")  # Forma esperada: (N, 13, 64)
    y = np.load(path + "y_test.npy")

    # Cargar el modelo guardado
    len_models = len(os.listdir(os.path.join(BASE_DIR, "model_weights")))
    model_path = os.path.join(BASE_DIR, "model_weights", f"voiceModel_{len_models-1}.keras")
    model = load_model(model_path)

    plot_confusion_matrix(model, X, y)