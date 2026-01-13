from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Primera capa: Convolución estándar para captar rasgos básicos
        layers.Conv2D(16, (3, 3), activation='leaky_relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Primera capa: Convolución estándar para captar rasgos básicos
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Segunda capa: Convolución separable (más ligera)
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Aplanar y Clasificar
        layers.Flatten(),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dropout(0.1),
        layers.Dense(32, activation='leaky_relu'),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax') # Probabilidad por clase
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model