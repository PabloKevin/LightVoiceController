import os
import librosa
import numpy as np
from tqdm import tqdm # Para ver la barra de progreso
from AudioProcessing import process_audio_wav

# --- Configuración ---
dataset = "Test" # Cambia a "Train" o "Test" según el conjunto que quieras procesar
DATA_PATH = f"/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_{dataset}Set_Augmented" # Carpeta con todos los .wav
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
    
def extract_fft_features(file_path, n_fft=256, hop_length=128, target_frames=64):
    """
    Carga un audio y calcula la magnitud de la STFT (FFT en el tiempo).
    n_fft: Tamaño de la ventana (determina la resolución en frecuencia).
    hop_length: Avance entre ventanas (determina la resolución en tiempo).
    """
    # 1. Cargar audio (sr=None mantiene la tasa de muestreo original, ej. 16k)
    audio, sr = librosa.load(file_path, sr=None)

    # Ajustar n_fft para garantizar 128 bins de frecuencia
    n_fft = 256 if n_fft % 2 == 0 else 255  # Asegurar que n_fft sea par

    # 2. Calcular la STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # 3. Obtener la magnitud (valor absoluto)
    magnitude = np.abs(stft)

    # 4. Convertir a escala logarítmica (Decibelios)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

    # 5. Ajustar dimensiones (Padding o Truncate) para tener siempre 'target_frames'
    if log_spectrogram.shape[1] > target_frames:
        log_spectrogram = log_spectrogram[:128, :target_frames]  # Truncar a 128 bins
    else:
        pad_width = target_frames - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram[:128, :], ((0, 0), (0, pad_width)), mode='constant')

    return log_spectrogram

def prepare_dataset():
    X = []
    y = []
    
    # Listar todos los archivos wav
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.wav') and f.startswith(("ambiente", "apagarLuz", "prenderLuz", "luzBaja", "luzMedia", "luzAlta"))]
    
    print(f"Iniciando extracción de MFCCs para {len(files)} archivos...")

    for filename in tqdm(files):
        # 1. Extraer la clase del nombre (Asume que el nombre es 'clase_numero_aug.wav')
        # Si tu formato es diferente, ajusta esta línea:
        label = filename.split('_')[0] 
        
        # 2. Obtener características
        full_path = os.path.join(DATA_PATH, filename)
        features = extract_fft_features(full_path)
        
        if features is not None:
            X.append(features)
            y.append(label)

    # Convertir a arrays de NumPy
    X = np.array(X)
    y = np.array(y)

    # 3. Guardar en disco
    np.save(f"/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/X_{dataset.lower()}.npy", X)
    np.save(f"/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/y_{dataset.lower()}.npy", y)
    
    print(f"\n¡Dataset listo!")
    print(f"Forma de X (Muestras, Coeficientes, Tiempo): {X.shape}")
    print(f"Clases detectadas: {np.unique(y)}")

if __name__ == "__main__":
    prepare_dataset()