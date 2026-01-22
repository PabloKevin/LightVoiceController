import numpy as np
import os
from scipy.io import wavfile
from MFCCsDataSets import extract_features
from tqdm import tqdm as progress_bar
import matplotlib.pyplot as plt
import librosa.display
from seaborn import boxplot

def map_Raw2MFCC(dataset = "Train", save=False):
    dir = f"RawAudio_{dataset}Set_Augmented/"
    files = [f for f in os.listdir(dir) if f.endswith('.wav') and f.startswith(("ambiente", "apagarLuz", "prenderLuz"))]
    X, Y = [], []
    for file in progress_bar(files):
        wavFile = dir+file
        fs, data = wavfile.read(wavFile)
        data = data.astype(np.float32)/(2**16/2)
        rawAudio = raw2windows(data)
        mfcc = extract_features(wavFile).T
        for i in range(len(rawAudio)):
            X.append(rawAudio[i])
            Y.append(mfcc[i])
    X = np.array(X)
    Y = np.array(Y)
    mean_yTrain = -47.319750827110255
    std_yTrain = 174.86826506622137
    Y = (Y - mean_yTrain) / std_yTrain

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


if __name__ == "__main__":
    X, Y = map_Raw2MFCC(dataset="Test", save=False)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    # Supongamos que X_train tiene forma (num_muestras, 375)
    #print(Y[0])

    plt.figure(figsize=(12, 6))
    boxplot(data=Y[0])
    plt.title('Distribución de todos los Coeficientes MFCC')
    plt.ylabel('Amplitud')
    plt.xlabel('Índice del Coeficiente')
    plt.xticks(rotation=45)
    plt.show()





