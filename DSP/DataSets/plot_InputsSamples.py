import numpy as np
import matplotlib.pyplot as plt

# Load the FFT dataset
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Select a sample to plot
sample_index = 540  # Change this to visualize other samples
fft_sample = X_train[sample_index]
print("sample shape:", fft_sample.shape)
label = y_train[sample_index]

# Plot the FFT features as a heatmap
plt.figure(figsize=(10, 6))
plt.title(f"FFT Features - Sample {sample_index} - Label: {label}")
plt.imshow(fft_sample, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label="Magnitude (dB)")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")
plt.show()
