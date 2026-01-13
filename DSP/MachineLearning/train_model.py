import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_arch import build_model
from pathlib import Path
import json
import os


def train_and_plot(X_train, y_train, X_val, y_val, num_classes):
    model = build_model((13, 64, 1), num_classes) # La función build_model que definimos antes
    
    # Entrenamiento
    print("\n🚀 Iniciando entrenamiento en GPU...")
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=64, # Puedes subirlo a 64 si tu GPU tiene mucha VRAM
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15) # Baja la velocidad si se estanca
        ],
        verbose=1
    )

    # --- Visualización de Métricas ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Gráfica de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.title('Precisión (Accuracy)')
    plt.xlabel('Época')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Gráfica de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title('Pérdida (Loss)')
    plt.xlabel('Época')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    return model

def train_test_split(X, y, test_size=0.2, random_state=47):
    # 1. Fijar la semilla para que los resultados sean reproducibles
    np.random.seed(random_state)
    
    # 2. Crear un array de índices y mezclarlos
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # 3. Calcular cuántos datos van a test
    test_count = int(X.shape[0] * test_size)
    
    # 4. Dividir los índices
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    # 5. Retornar los arrays divididos
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    # Configuración para evitar errores de memoria en la GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detectada: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("No se detectó GPU. Se usará la CPU.")

    # 1. Cargar tus archivos generados
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "DataSets/")
    X = np.load(path + "X_train.npy")  # Forma esperada: (N, 13, 64)
    y = np.load(path + "y_train.npy")

    # 2. Codificar etiquetas
    # Cargar el archivo JSON
    with open('class_map.json', 'r') as file:
        mapping = json.load(file)

    # Acceder a los diccionarios
    CLASS_MAP = mapping["CLASS_MAP"]
    INDEX_TO_CLASS = mapping["INDEX_TO_CLASS"]

    # 2. Mapeo de etiquetas a indices
    y_encoded = np.array([CLASS_MAP[label] for label in y])
    num_classes = len(np.unique(y_encoded))

    # 3. Añadir la dimensión de canal y Normalizar
    # NORMALIZACIÓN CRÍTICA
    # Calculamos media y desviación sobre todo el dataset
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / (std + 1e-8) # El 1e-8 evita división por cero

    # Ahora sí, el reshape
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # 4. Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    input_shape = (13, 64, 1)
    model = build_model(input_shape, num_classes)
    model.summary()

    model = train_and_plot(X_train, y_train, X_val, y_val, num_classes)

    # Guardar el modelo completo (Arquitectura + Pesos + Optimización)
    output_path = BASE_DIR + "/model_weights/"
    model_name = output_path + "voiceModel_0.keras"
    while Path(model_name).exists():
        base, ext = model_name.rsplit('_', 1)
        number = int(ext.split('.')[0]) + 1
        model_name = f"{base}_{number}.keras"

    model.save(model_name) 
    print("Modelo guardado correctamente como:", model_name)

    