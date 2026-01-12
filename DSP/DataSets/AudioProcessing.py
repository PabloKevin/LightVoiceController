import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import medfilt
import librosa
import librosa.display

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Crea los coeficientes del filtro Butterworth.
    lowcut: Frecuencia mínima
    highcut: Frecuencia máxima
    fs: Frecuencia de muestreo (16000 en tu caso)
    order: "Fuerza" del filtro (5 es un buen equilibrio)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def bandstop_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica un filtro que ELIMINA las frecuencias entre lowcut y highcut.
    lowcut: Inicio de la banda a eliminar (Hz)
    highcut: Fin de la banda a eliminar (Hz)
    fs: Frecuencia de muestreo (16000 Hz)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # btype='bandstop' es la clave aquí
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

def highpass_filter(data, cutoff, fs, order=5):
    """
    Aplica un filtro que ELIMINA las frecuencias entre lowcut y highcut.
    lowcut: Inicio de la banda a eliminar (Hz)
    highcut: Fin de la banda a eliminar (Hz)
    fs: Frecuencia de muestreo (16000 Hz)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # btype='bandstop' es la clave aquí
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y
    

def plotSignal(data, sample_rate, filename):
    # Crear el eje del tiempo en segundos
    length = data.shape[0] / sample_rate
    time = np.linspace(0., length, data.shape[0])

    plt.figure(figsize=(12, 4))
    plt.plot(time, data, color='blue')
    
    plt.title(f"Visualización de: {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud (16-bit PCM)")
    plt.grid(True)
    
    # Dibujar línea en el cero para ver el offset
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.show()



def plot_frequency_spectrum(data, fs, title="Espectro de Frecuencia"):
    """
    data: El array de audio (muestras)
    fs: Frecuencia de muestreo (ej. 16000)
    """
    n = len(data)
    
    # Calcular la FFT
    # Aplicamos una ventana de Hann para evitar el "leakage" espectral
    window = np.hanning(n)
    yf = fft(data * window)
    
    # Obtener las frecuencias correspondientes
    xf = fftfreq(n, 1 / fs)

    # Solo nos interesa la primera mitad (frecuencias positivas)
    # y calculamos la magnitud en decibelios (opcional) o absoluta
    pos_mask = xf >= 0
    xf = xf[pos_mask]
    yf_magnitude = np.abs(yf[pos_mask])

    plt.figure(figsize=(12, 5))
    plt.plot(xf, yf_magnitude, color='red')
    
    plt.title(title)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid(True)
    
    # Si quieres ver mejor la voz humana, limita el eje X a 4000Hz
    #plt.xlim(0, 800) 
    
    plt.show()

def gaussian_blur(data, window_size=7):
    # Creamos una ventana de suavizado suave (tipo campana)
    window = np.hamming(window_size)
    window /= window.sum()
    
    # Aplicamos la convolución
    data_suave = np.convolve(data, window, mode='same')
    return data_suave

def audio_blur(data, window_size=5):
    """
    Aplica una convolución (blur) a la señal.
    window_size: qué tan fuerte es el blur. (3-11 es lo ideal)
    """
    kernel = np.ones(window_size) / window_size
    # mode='same' mantiene la señal del mismo tamaño
    return np.convolve(data, kernel, mode='same')

from scipy.signal import butter, sosfilt, sosfreqz

def apply_bandstop_stable(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Usamos 'sos' (Second-Order Sections) en lugar de 'ba'
    sos = butter(order, [low, high], btype='bandstop', output='sos')
    
    # Aplicamos sosfilt en lugar de lfilter
    y = sosfilt(sos, data)
    return y

def plot_mfcc(wav_file, n_mfcc=13, fs=12000):
    # 1. Cargar el audio
    # sr=None mantiene la frecuencia original del archivo
    y, sr = librosa.load(wav_file, sr=fs)

    # 2. Calcular los MFCCs
    # n_mfcc: cantidad de coeficientes (para ESP32, entre 10 y 20 es ideal)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3. Visualización
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCCs - {wav_file}')
    plt.ylabel('Coeficientes MFCC')
    plt.xlabel('Tiempo')
    plt.tight_layout()
    plt.show()


def process_audio_wav(wav_path, output_path=None, plot=False):
    wavFile = wav_path
    fs, data = wavfile.read(wavFile)

    data = data/np.max(np.abs(data))
    data = data - np.mean(data)  # Eliminar offset DC
    data = data[len(data)//15:]

    data = audio_blur(data, window_size = 5)
    data = gaussian_blur(data, window_size = 13)
    data = medfilt(data, kernel_size=59) #

    # 2. Aplicar filtro (Frecuencias para voz humana: 300-3000Hz)
    data_filtrada = bandpass_filter(data, 320.0, 3000.0, fs, order=6)[len(data)//32 : -len(data)//256]
    data_filtrada = highpass_filter(data_filtrada, 340.0, fs, order=7)

    data_filtrada = apply_bandstop_stable(data_filtrada, 380, 500, fs)

    if plot:
        #data_dB = 20 * np.log10(np.abs(data_filtrada))
        plotSignal(data_filtrada, fs, wavFile.split("/")[-1]+"_Filtered")
        plot_frequency_spectrum(data_filtrada, fs, title="Espectro de Frecuencia - Audio Filtrado")

    if output_path is not None:
        # 3. Guardar el resultado limpio y normalizado en int16
        data_filtrada_int16 = np.int16(data_filtrada / np.max(np.abs(data_filtrada)) * 32767).astype(np.int16)
        wavfile.write(output_path+wavFile.split("/")[-1], fs, data_filtrada_int16)
        print("Filtro aplicado con éxito.")
        plot_mfcc(output_path+wavFile.split("/")[-1])
    
    return data_filtrada, fs

if __name__ == "__main__":
    wav_file = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_TestSet_Augmented/luzAlta_7_aug_2.wav"
    output_path = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/ProcessedAudio/"
    process_audio_wav(wav_file, output_path, plot=True)