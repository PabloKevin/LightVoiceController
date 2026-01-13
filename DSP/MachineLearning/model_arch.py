from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # --- BLOQUE CONVOLUCIONAL (Extractor) ---
        # input_shape: (13, 64, 1) -> (MFCCs, Tiempo, Canal)
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((1, 2)), # Reducimos solo la dimensión temporal para mantener MFCCs
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((1, 2)), # Reducimos solo la dimensión temporal para mantener MFCCs
        layers.Dropout(0.2),

        # --- PREPARACIÓN PARA RNN ---
        # Necesitamos pasar de 4D (Batch, F, T, C) a 3D (Batch, T, Features)
        # Reshape dinámico basado en lo que salga de la CNN
        layers.Reshape((-1, 64 * 13)), # Colapsamos frecuencias y canales en un vector por paso de tiempo

        # --- BLOQUE RECURRENTE (Memoria temporal) ---
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.3),

        # --- CLASIFICADOR ---
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

class voiceModelNN():
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, X):
        mean, std = -50.281225, 159.534246
        X = (X - mean) / (std + 1e-8)
        if len(X.shape) == 4:
            X = X.reshape(X.shape[0], 13, 64, 1)
        
        return self.model.predict(X)