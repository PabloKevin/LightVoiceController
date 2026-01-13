import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model_arch import build_model
from pathlib import Path
import json
import os

# --- FUNCIONES DE SOPORTE PARA CROSS VALIDATION ---

def get_kfold_indices(X, k=5, random_state=42):
    """Divide los índices en K grupos aleatorios sin usar sklearn."""
    indices = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    return np.array_split(indices, k)

def plot_cv_results(all_histories):
    """Grafica las métricas manejando folds de distintas longitudes."""
    plt.figure(figsize=(12, 5))
    
    # Determinar la longitud máxima de épocas
    max_epochs = max(len(h['val_accuracy']) for h in all_histories)
    
    # --- Gráfica de Accuracy ---
    plt.subplot(1, 2, 1)
    acc_matrix = []
    for i, h in enumerate(all_histories):
        val_acc = h['val_accuracy']
        plt.plot(val_acc, alpha=0.3, label=f'Fold {i+1}')
        
        # Rellenar con el último valor para poder calcular el promedio
        padded = np.pad(val_acc, (0, max_epochs - len(val_acc)), 'edge')
        acc_matrix.append(padded)
    
    avg_acc = np.mean(acc_matrix, axis=0)
    plt.plot(avg_acc, color='blue', linewidth=2, label='Promedio Val')
    plt.title('Precisión (Accuracy) entre Folds')
    plt.legend()
    plt.grid(True)

    # --- Gráfica de Loss ---
    plt.subplot(1, 2, 2)
    loss_matrix = []
    for h in all_histories:
        val_loss = h['val_loss']
        padded_loss = np.pad(val_loss, (0, max_epochs - len(val_loss)), 'edge')
        loss_matrix.append(padded_loss)
        
    avg_loss = np.mean(loss_matrix, axis=0)
    plt.plot(avg_loss, color='red', linewidth=2, label='Promedio Loss')
    plt.title('Pérdida (Loss) Promedio')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# --- BLOQUE PRINCIPAL ---

if __name__ == "__main__":
    # Configuración GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

    # 1. Cargar Datos
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "DataSets/")
    X = np.load(os.path.join(data_path, "X_train.npy"))
    y = np.load(os.path.join(data_path, "y_train.npy"))

    with open('class_map.json', 'r') as file:
        mapping = json.load(file)
    CLASS_MAP = mapping["CLASS_MAP"]
    y_encoded = np.array([CLASS_MAP[label] for label in y])
    num_classes = len(CLASS_MAP)

    # 2. Normalización y Reshape
    mean, std = np.mean(X), np.std(X)
    print(f"Normalizando datos: mean={mean:.6f}, std={std:.6f}")

    

    # 3. Configuración de Cross-Validation
    K = 5
    folds = get_kfold_indices(X, k=K)
    all_histories = []
    input_shape = (13, 64, 1)

    print(f"\n🔄 Iniciando {K}-Fold Cross Validation...")

    best_accuracy = -1.0
    for i in range(K):
        print(f"\n--- ENTRENANDO FOLD {i+1}/{K} ---")
        
        # Crear conjuntos de Train y Val para este Fold
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

        mean, std = np.mean(X_train_fold), np.std(X_train_fold)
        X_train_fold = (X_train_fold - mean) / (std + 1e-8)
        X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], 13, 64, 1)
        X_val_fold = (X_val_fold - mean) / (std + 1e-8)
        X_val_fold = X_val_fold.reshape(X_val_fold.shape[0], 13, 64, 1)

        # Re-construir el modelo desde cero para cada fold (evita contaminación)
        model = build_model(input_shape, num_classes)
        
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=250, # Bajé a 100 porque con CV el tiempo total se multiplica por K
            batch_size=64,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=17)
            ],
            verbose=0 # Reducir ruido en consola
        )
        
        all_histories.append(history.history)
        print(f"Fold {i+1} completado. Val Acc: {max(history.history['val_accuracy']):.4f}")

        val_acc_history = history.history['val_accuracy']
        current_fold_acc = max(val_acc_history)
        
        if current_fold_acc > best_accuracy:
            best_accuracy = current_fold_acc
            output_path = os.path.join(BASE_DIR, "model_weights")
            model.save(os.path.join(output_path, "voiceModel_BEST.keras"))
            print(f"--> Guardado como mejor modelo (Fold {i+1})")

    # 4. Visualización de resultados globales
    plot_cv_results(all_histories)
    print("\n✅ Proceso completado. Resultados de CV analizados.")
