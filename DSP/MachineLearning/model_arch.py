from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Primera capa: Convolución estándar para captar rasgos básicos
        layers.SeparableConv2D(256, (5, 5), activation='leaky_relu', input_shape=input_shape, padding='same'),
        layers.AvgPool2D((2, 2)),
        layers.Dropout(0.2),

        layers.SeparableConv2D(256, (3, 3), activation='leaky_relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Primera capa: Convolución estándar para captar rasgos básicos
        layers.SeparableConv2D(512, (2, 2), activation='leaky_relu', padding='same'),
        layers.Dropout(0.4),


        # Aplanar y Clasificar
        layers.Flatten(),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='leaky_relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') # Probabilidad por clase
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