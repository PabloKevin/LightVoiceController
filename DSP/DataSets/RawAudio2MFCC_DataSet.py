import numpy as np
import os
from scipy.io import wavfile
from MFCCsDataSets import extract_features
from tqdm import tqdm as progress_bar
import matplotlib.pyplot as plt
import librosa.display
from seaborn import boxplot
from AudioProcessing import process_audio_wav

def map_Raw2MFCC(dataset = "Train", windows=64, save=False):
    dir = f"RawAudio_{dataset}Set_Augmented/"
    files = [f for f in os.listdir(dir) if f.endswith('.wav') and f.startswith(("ambiente", "apagarLuz", "prenderLuz"))]
    X, Y = [], []
    for file in progress_bar(files):
        wavFile = dir+file
        fs, data = wavfile.read(wavFile)
        data = data.astype(np.float32)/(2**16/2)
        rawAudio = raw2windows(data, windows)
        mfcc = extract_features(wavFile, windows).T
        for i in range(len(rawAudio)):
            X.append(rawAudio[i])
            Y.append(mfcc[i])

    X = np.array(X)
    Y = np.array(Y)

    if save:
        np.save(f"RawAudio2MFCC/X_{dataset.lower()}.npy", X)
        np.save(f"RawAudio2MFCC/Y_{dataset.lower()}.npy", Y)

    return X, Y


def map_Raw2Processed(dataset = "Train", windows=64, save=False):
    dir = f"RawAudio_{dataset}Set_Augmented/"
    files = [f for f in os.listdir(dir) if f.endswith('.wav') and f.startswith(("ambiente", "apagarLuz", "prenderLuz"))]
    X, Y = [], []

    for file in progress_bar(files):
        wavFile = dir+file
        fs, data = wavfile.read(wavFile)
        data = data.astype(np.float32)/(2**16/2)
        rawAudio = raw2windows(data, windows)
        processedAudio, fs = process_audio_wav(wavFile)
        processedAudio = processed2windows(processedAudio, windows)
        for i in range(len(rawAudio)):
            X.append(rawAudio[i])
            Y.append(processedAudio[i])

    X = np.array(X)
    Y = np.array(Y)

    if save:
        np.save(f"RawAudio2MFCC/X_{dataset.lower()}.npy", X)
        np.save(f"RawAudio2MFCC/Y_{dataset.lower()}.npy", Y)

    return X, Y

def raw2windows(rawAudio, windows=64):
    window_len = len(rawAudio)// windows
    raw_array = []
    for i in range(windows):
        raw_array.append(rawAudio[window_len*i : window_len + window_len*i])
    return np.array(raw_array)

def processed2windows(processedAudio, windows=64):
    dif = len(processedAudio) - 310*64
    if dif<0:
        dif*=-1
        processedAudio = np.pad(processedAudio, (0, dif), mode='constant', constant_values=0)
    elif dif>0:
        processedAudio = processedAudio[dif:]

    window_len = len(processedAudio)// windows
    raw_array = []
    for i in range(windows):
        raw_array.append(processedAudio[window_len*i : window_len + window_len*i])
    return np.array(raw_array)


if __name__ == "__main__":
    #X, Y = map_Raw2MFCC(dataset="Test", windows=64, save=False)
    X, Y = map_Raw2Processed(dataset="Train", windows=32, save=True)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Xmean={np.mean(X)}, Xstd={np.std(X)}")
    print(f"Ymean={np.mean(Y)}, Ystd={np.std(Y)}")
    # Supongamos que X_train tiene forma (num_muestras, 375)

    """plt.figure(figsize=(12, 6))
    boxplot(data=Y[0])
    plt.title('Distribución de todos los Coeficientes MFCC')
    plt.ylabel('Amplitud')
    plt.xlabel('Índice del Coeficiente')
    plt.xticks(rotation=45)
    plt.show()"""





