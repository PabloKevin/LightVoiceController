from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import numpy as np

def build_model(input_shape, output_shape):
    model = models.Sequential([
        # --- ENCODER ---
        layers.Input(shape=input_shape),
        # Extrae características y reduce el ruido
        layers.Conv1D(32, kernel_size=11, activation='leaky_relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.MaxPooling1D(5), # Reducción fuerte para captar la forma global
        layers.Dropout(0.2),
        
        # --- BOTTLENECK ---
        layers.Conv1D(128, kernel_size=3, activation='leaky_relu', padding='same'),
        
        # --- DECODER ---
        # Reconstruye la señal a su tamaño original
        layers.UpSampling1D(5),
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.UpSampling1D(2),
        layers.Conv1D(32, kernel_size=11, activation='leaky_relu', padding='same'),
        layers.Dropout(0.2),
        
        # --- SALIDA ---
        # 1 filtro para reconstruir la onda mono
        layers.Conv1D(1, kernel_size=3, activation='linear', padding='same'),
        layers.Flatten(),
        layers.Dense(32, activation="leaky_relu"),
        layers.Dense(output_shape)
    ])
    

    # El valor por defecto de Adam es 0.001
    opt = optimizers.Adam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-07
    )
    model.compile(optimizer=opt, loss='log_cosh', metrics=['mae'])
    return model

class voiceModelNN():
    def __init__(self, model_path, X_train=None, mean=-50.281225, std=159.534246):
        self.model = load_model(model_path)
        if X_train is not None:
            self.mean = np.mean(X_train)
            self.std = np.std(X_train)
        else:
            self.mean=mean
            self.std=std
        

    def predict(self, X):
        X = (X - self.mean) / (self.std + 1e-8)
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], 13, 64, 1)
            print(f"Input reshaped to: {X.shape}")
        
        return self.model.predict(X)