import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

def butter_bandpass(lowcut, highcut, fs, order=5):
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
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica el filtro a un array de datos.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
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
    #plt.xlim(0, 4000) 
    
    plt.show()

def gaussian_blur(data, window_size=7):
    # Creamos una ventana de suavizado suave (tipo campana)
    window = np.hamming(window_size)
    window /= window.sum()
    
    # Aplicamos la convolución
    data_suave = np.convolve(data, window, mode='same')
    return data_suave.astype(np.int16)

# 1. Leer el archivo que grabaste
wavFile = "RawAudioDataSet/recording_12.wav"
fs, data = wavfile.read(wavFile)

data_blur = gaussian_blur(data, window_size = 7)

# 2. Aplicar filtro (Frecuencias para voz humana: 300-3000Hz)
data_filtrada = apply_bandpass_filter(data_blur, 250.0, 4000.0, fs, order=6)[len(data)//128 : -len(data)//256]

# 3. (Opcional) Normalizar el audio para que se escuche más fuerte
data_filtrada = np.int16(data_filtrada / np.max(np.abs(data_filtrada)) * 32767)

# 4. Guardar el resultado limpio
wavfile.write("ProcessedAudioDataSet/"+wavFile.split("/")[-1], fs, data_filtrada)
print("Filtro aplicado con éxito.")

plotSignal(data_filtrada, fs, wavFile.split("/")[-1]+"_Filtered")
plot_frequency_spectrum(data_filtrada, fs, title="Espectro de Frecuencia - Audio Filtrado")