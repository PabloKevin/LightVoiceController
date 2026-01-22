from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import numpy as np

def build_model(input_shape, output_shape):
    model = models.Sequential([
        # 1. Capa de entrada
        layers.Input(shape=input_shape),
        
        # 2. (Opcional) Una Conv1D para reducir la carga de la LSTM
        # Ayuda a extraer rasgos locales antes de pasar a la memoria temporal
        #layers.Conv1D(filters=32, kernel_size=5, strides=2, activation='relu', padding='same'),
        #layers.MaxPooling1D(pool_size=2),
        #layers.Conv1D(filters=64, kernel_size=5, strides=2, activation='relu', padding='same'),
        
        # 3. Capa LSTM
        # 'units=64' es un buen balance entre poder y memoria para el ESP32
        # return_sequences=False porque solo queremos la predicción final del frame
        layers.LSTM(units=128, return_sequences=False),
        
        # 4. Capas Densas para mapear la memoria a los coeficientes
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), # Ayuda a evitar que el modelo memorice ruidos
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2), # Ayuda a evitar que el modelo memorice ruidos
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2), # Ayuda a evitar que el modelo memorice ruidos
        
        # 5. Salida (13 coeficientes MFCC)
        layers.Dense(output_shape, activation='linear')
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