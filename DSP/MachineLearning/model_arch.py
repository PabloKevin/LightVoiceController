from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # --- BLOQUE CONVOLUCIONAL (Extractor) ---
        # input_shape: (13, 64, 1) -> (MFCCs, Tiempo, Canal)
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dropout(0.05),
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same', input_shape=input_shape, strides=1),
        layers.MaxPooling2D((2, 1)), # Reducimos solo la dimensión temporal para mantener MFCCs
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same', strides=1),
        layers.MaxPooling2D((2, 2)), # Reducimos solo la dimensión temporal para mantener MFCCs
        layers.Dropout(0.4),
        layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same', strides=1),
        #layers.MaxPooling2D((3, 1)), # Reducimos solo la dimensión temporal para mantener MFCCs
        layers.Dropout(0.4),

        # --- PREPARACIÓN PARA RNN ---
        # Necesitamos pasar de 4D (Batch, F, T, C) a 3D (Batch, T, Features)
        # --- ELIMINAR DIMENSIÓN 0 ---
       
        #layers.Reshape((32, 64)),

        layers.Flatten(),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dropout(0.35),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
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
        #X = (X - self.mean) / (self.std + 1e-8)
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], 13, 64, 1)
            print(f"Input reshaped to: {X.shape}")
        
        return self.model.predict(X)