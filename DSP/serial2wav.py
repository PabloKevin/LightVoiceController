import serial
import wave
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# --- Configuración ---
SERIAL_PORT = '/dev/ttyUSB0'  # Cambia esto según tu puerto 
BAUD_RATE = 115200
SAMPLE_RATE = 12000           # Debe coincidir con el ESP32
TIME_TO_RECORD = 2 #Segundos
OUTPUT_PATH = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_pruebas/"
SAMPLES_TO_READ = SAMPLE_RATE * TIME_TO_RECORD       # 16000 Hz * 2 segundos

OUTPUT_FILE = OUTPUT_PATH + "apagarLuz_0.wav"
while Path(OUTPUT_FILE).exists():
    base, ext = OUTPUT_FILE.rsplit('_', 1)
    number = int(ext.split('.')[0]) + 1
    OUTPUT_FILE = f"{base}_{number}.wav"


def save_as_wav(data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)          # Monoaural
        wf.setsampwidth(2)         # 16 bits (2 bytes por muestra)
        wf.setframerate(SAMPLE_RATE)
        
        # Convertimos los enteros a bytes binarios (formato 'h' es short/16bit)
        binary_data = struct.pack('<' + ('h' * len(data)), *data)
        wf.writeframes(binary_data)
    print(f"Archivo guardado como: {filename}")

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        print(f"Conectado a {SERIAL_PORT}. Esperando audio...")
        
        # Enviamos 'g' para que el ESP32 empiece a grabar y transmitir
        ser.write(b'g')
        
        audio_data = []
        count = 0
        line = ""
        while count < SAMPLES_TO_READ and line!="Finish":
            line = ser.readline().decode('ascii', errors='ignore').strip()
            
            try:
                # Esto aceptará números positivos y negativos (ej: "120", "-45")
                val = int(line) 
                audio_data.append(val)
                count += 1
                if count % 1000 == 0:
                    print(f"Recibidas {count}/{SAMPLES_TO_READ} muestras...")
            except ValueError:
                # Si la línea es texto (como ">>> Iniciando..."), la ignoramos
                if line:
                    print(f"Texto recibido (no audio): {line}")

        save_as_wav(audio_data, OUTPUT_FILE)
        ser.close()

    except Exception as e:
        print(f"Error: {e}")



def plot_wav(filename):
    # Leer el archivo
    sample_rate, data = wavfile.read(filename)
    
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

if __name__ == "__main__":
    main()
    #plot_wav(OUTPUT_FILE)