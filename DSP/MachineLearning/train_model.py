import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_arch import build_model
import json
import os

if __name__ == "__main__":
    # --- Configuración GPU ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

    # --- 1. Cargar Datos ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "DataSets/")
    X = np.load(os.path.join(data_path, "X_train.npy"))
    y = np.load(os.path.join(data_path, "y_train.npy"))

    with open('class_map.json', 'r') as file:
        mapping = json.load(file)
    CLASS_MAP = mapping["CLASS_MAP"]
    y_encoded = np.array([CLASS_MAP[label] for label in y])
    num_classes = len(CLASS_MAP)


    # --- 3. Entrenamiento ---
    input_shape = (13, 64, 1)
    model = build_model(input_shape, num_classes)
    
    # Entrenar por un número fijo de épocas 
    # (Usa el número de épocas donde viste que el modelo convergía en tus pruebas previas)
    EPOCHS = 250 
    
    print(f"\n🚀 Entrenando durante {EPOCHS} épocas...")
    history = model.fit(
        X, y_encoded,
        epochs=EPOCHS,
        batch_size=128,
        verbose=1,
        # Callback para reducir el LR si la pérdida de entrenamiento se estanca
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=13, verbose=1)
        ]
    )

    # --- 4. Guardar Modelo y Estadísticas ---
    output_path = os.path.join(BASE_DIR, "model_weights")
    os.makedirs(output_path, exist_ok=True)
    
    model_name = os.path.join(output_path, "voiceModel_FullTrain.keras")
    model.save(model_name)
    
    # Guardamos los valores de normalización en un JSON para el ESP32
    norm_values = {
        "mean": float(global_mean),
        "std": float(global_std)
    }
    with open(os.path.join(output_path, "norm_params.json"), "w") as f:
        json.dump(norm_values, f)

    print(f"\n✅ Proceso completado.")
    print(f"Modelo: {model_name}")
    print(f"Parámetros de normalización guardados en norm_params.json")

    # --- 5. Gráficas de Entrenamiento ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.title('Precisión de Entrenamiento')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Pérdida de Entrenamiento')
    plt.show()