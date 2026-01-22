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
    data_path = os.path.join(BASE_DIR, "../..", "DataSets/RawAudio2MFCC")
    X = np.load(os.path.join(data_path, "X_train.npy"))
    y = np.load(os.path.join(data_path, "Y_train.npy"))

    # --- 3. Entrenamiento ---
    input_shape = (375, 1)
    output_shape = 13
    model = build_model(input_shape, output_shape)
    
    # Entrenar por un número fijo de épocas 
    # (Usa el número de épocas donde viste que el modelo convergía en tus pruebas previas)
    EPOCHS = 300 
    
    print(f"\n🚀 Entrenando durante {EPOCHS} épocas...")
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=256,
        verbose=1,
        # Callback para reducir el LR si la pérdida de entrenamiento se estanca
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=13, verbose=1)
        ]
    )

    # --- 4. Guardar Modelo y Estadísticas ---
    output_path = os.path.join(BASE_DIR, "model_weights")
    os.makedirs(output_path, exist_ok=True)
    
    model_name = os.path.join(output_path, "audioProcessingModel.keras")
    model.save(model_name)

    print(f"\n✅ Proceso completado.")
    print(f"Modelo: {model_name}")
    print(f"Parámetros de normalización guardados en norm_params.json")

    # --- 5. Gráficas de Entrenamiento ---
    plt.figure(figsize=(12, 5))
    # Gráfica de MAE (Error absoluto medio)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.title('Error Medio (MAE)')
    plt.legend()

    # Gráfica de Pérdida (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Pérdida (Loss/MSE)')
    plt.legend()
    plt.show()