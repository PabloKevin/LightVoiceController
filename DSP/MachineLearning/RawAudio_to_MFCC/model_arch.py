from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np

def build_model(input_shape, output_shape):
    model = models.Sequential([
        # Entrada: 375 muestras de audio crudo (1 solo canal)
        layers.Input(shape=input_shape),
        
        # Primera capa convolucional: aprende filtros básicos
        layers.Conv1D(filters=16, kernel_size=5, activation='leaky_relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(filters=32, kernel_size=9, activation='leaky_relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Segunda capa convolucional: patrones más complejos
        layers.Conv1D(filters=64, kernel_size=5, activation='leaky_relu', padding='same'),
        #layers.GlobalAveragePooling1D(), # Reduce el mapa de características a un vector
        layers.MaxPooling1D(pool_size=4),
        layers.Flatten(),
        
        # Capas densas de razonamiento
        layers.Dense(256, activation='leaky_relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(64, activation='leaky_relu'),
        layers.Dropout(0.2),
        
        # Capa de salida: 13 coeficientes MFCC (Linear)
        layers.Dense(output_shape) 
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
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