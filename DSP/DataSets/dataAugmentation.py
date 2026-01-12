import numpy as np
import os
import wave
from scipy.io import wavfile
from pathlib import Path
from joblib import Parallel, delayed

# --- Configuración ---
FS = 12000  # Tu frecuencia de 12kHz
INPUT_DIR = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_TestSet"
OUTPUT_DIR = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_TestSet_Augmented"
AUGMENTATIONS_PER_FILE = 10  # Cuántas variaciones crear por cada audio original

def load_wav(path):
    fs, data = wavfile.read(path)
    # Asegurarnos de que sea float32 para procesar
    return data.astype(np.float32)

def save_wav(path, data, fs):
    # Normalizar y convertir a int16 para el formato WAV estándar
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    data = (data * 32767).astype(np.int16)
    wavfile.write(path, fs, data)

# --- Funciones de Aumento ---

def change_speed(data, fs, rate_range=(0.85, 1.15)):
    rate = np.random.uniform(*rate_range)
    input_length = len(data)
    # Interpolación para cambiar velocidad sin cambiar el tono drásticamente
    new_indices = np.linspace(0, input_length - 1, int(input_length / rate))
    new_data = np.interp(new_indices, np.arange(input_length), data)
    
    # Ajustar al tamaño original (Padding o Cut)
    if len(new_data) > input_length:
        return new_data[:input_length]
    else:
        return np.pad(new_data, (0, input_length - len(new_data)), 'constant')

def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise * np.max(np.abs(data))

def shift_time(data, max_shift_ms=150):
    # Desplazar el audio circularmente para simular desfase en el inicio
    shift_samples = int((max_shift_ms / 1000) * FS)
    shift = np.random.randint(-shift_samples, shift_samples)
    return np.roll(data, shift)

def change_volume(data, factor_range=(0.6, 1.4)):
    return data * np.random.uniform(*factor_range)

# --- Proceso Principal ---

def process_single_file(file_path, output_subfolder):
    filename = Path(file_path).stem
    data = load_wav(file_path)
    
    for i in range(AUGMENTATIONS_PER_FILE):
        # Aplicar cadena de aumentos aleatorios
        aug_data = change_speed(data, FS)
        aug_data = shift_time(aug_data)
        aug_data = add_noise(aug_data, noise_level=np.random.uniform(0.001, 0.008))
        aug_data = change_volume(aug_data)
        
        output_name = f"{filename}_aug_{i}.wav"
        save_wav(os.path.join(output_subfolder, output_name), aug_data, FS)

def run_augmentation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.wav')]
    
    print(f"Aumentando DataSet: ({len(files)} archivos)...")
    
    # Procesamiento en paralelo para ir rápido
    Parallel(n_jobs=-1)(delayed(process_single_file)(f, OUTPUT_DIR) for f in files)

if __name__ == "__main__":
    run_augmentation()
    print("¡Data Augmentation completado!")