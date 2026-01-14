import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from model_arch import voiceModelNN

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

    calculate_metrics(y_idx, y_pred_classes, len(classes))

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

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calcula las métricas de accuracy, precision, recall y f1-score.

    Args:
        y_true (list or np.array): Etiquetas reales.
        y_pred (list or np.array): Etiquetas predichas.
        num_classes (int): Número total de clases.

    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    # Inicializar variables
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)
    total_samples = len(y_true)

    # Calcular TP, FP y FN para cada clase
    for i in range(num_classes):
        true_positives[i] = np.sum((y_true == i) & (y_pred == i))
        false_positives[i] = np.sum((y_true != i) & (y_pred == i))
        false_negatives[i] = np.sum((y_true == i) & (y_pred != i))

    # Calcular métricas por clase
    precision = np.divide(true_positives, true_positives + false_positives, out=np.zeros_like(true_positives), where=(true_positives + false_positives) != 0)
    recall = np.divide(true_positives, true_positives + false_negatives, out=np.zeros_like(true_positives), where=(true_positives + false_negatives) != 0)
    f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)

    # Calcular métricas globales (ponderadas por el soporte de cada clase)
    accuracy = np.sum(y_true == y_pred) / total_samples
    weighted_precision = np.sum(precision * (true_positives + false_negatives)) / total_samples
    weighted_recall = np.sum(recall * (true_positives + false_negatives)) / total_samples
    weighted_f1_score = np.sum(f1_score * (true_positives + false_negatives)) / total_samples

    metrics_dict = {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1_score
    }
    print(metrics_dict)

    return metrics_dict

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "DataSets/")
    X = np.load(path + "X_test.npy")  # Forma esperada: (N, 13, 64)
    y = np.load(path + "y_test.npy")

    # Cargar el modelo guardado
    len_models = len(os.listdir(os.path.join(BASE_DIR, "model_weights")))
    #model_path = os.path.join(BASE_DIR, "model_weights", f"voiceModel_{len_models-1}.keras")
    model_path = os.path.join(BASE_DIR, "model_weights", f"voiceModel_BEST.keras")
    model = voiceModelNN(model_path)

    plot_confusion_matrix(model, X, y)