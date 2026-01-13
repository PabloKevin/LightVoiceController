import os
import librosa
import numpy as np
from tqdm import tqdm # Para ver la barra de progreso
from AudioProcessing import process_audio_wav

# --- Configuración ---
DATA_PATH = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_TestSet_Augmented" # Carpeta con todos los .wav
FS = 12000                      # Tu frecuencia de 12kHz
N_MFCC = 13                     # Número de coeficientes (ideal para ESP32)
MAX_LEN = 64                    # Longitud temporal fija 

def extract_features(file_path):
    try:
        # Cargar audio
        #y, sr = librosa.load(file_path, sr=FS)
        y, sr = process_audio_wav(file_path)
        
        # Extraer MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        
        # Ajustar tamaño para que todos los ejemplos sean iguales (Padding/Truncate)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        return mfcc
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None

def prepare_dataset():
    X = []
    y = []
    
    # Listar todos los archivos wav
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.wav')]
    
    print(f"Iniciando extracción de MFCCs para {len(files)} archivos...")

    for filename in tqdm(files):
        # 1. Extraer la clase del nombre (Asume que el nombre es 'clase_numero_aug.wav')
        # Si tu formato es diferente, ajusta esta línea:
        label = filename.split('_')[0] 
        
        # 2. Obtener características
        full_path = os.path.join(DATA_PATH, filename)
        features = extract_features(full_path)
        
        if features is not None:
            X.append(features)
            y.append(label)

    # Convertir a arrays de NumPy
    X = np.array(X)
    y = np.array(y)

    # 3. Guardar en disco
    np.save("/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/X_test.npy", X)
    np.save("/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/y_test.npy", y)
    
    print(f"\n¡Dataset listo!")
    print(f"Forma de X (Muestras, Coeficientes, Tiempo): {X.shape}")
    print(f"Clases detectadas: {np.unique(y)}")

if __name__ == "__main__":
    prepare_dataset()