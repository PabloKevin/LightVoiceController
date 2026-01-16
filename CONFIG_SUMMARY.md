# Configuration Summary

## TFLite Model
- **File**: `DSP/MachineLearning/model_weights/tflite_models/voiceModel_3classes.tflite`
- **Size**: 204 KB
- **Embedded in**: `include/model_data.h`
- **Classes**: 3 (ambiente, apagarLuz, prenderLuz)

## Normalization Parameters
```
Mean: -47.3197508271
Std:  174.8682650662
```

## Audio Configuration
```
Sample Rate:     12,000 Hz
Record Duration: 2 seconds
Total Samples:   24,000
ADC Pin:         34
Resolution:      12-bit
```

## MFCC Configuration
```
N_FFT:          512
HOP_LENGTH:     160 samples (~13.3 ms)
N_MELS:         40 mel-bands
N_MFCC:         13 coefficients
MAX_TIME_STEPS: 64 frames
```

## Filter Coefficients (Butterworth)

### Bandpass 320-3000 Hz (Order 6)
```
b = [0.00000033, 0, -0.00000330, 0, 0.00000825, ...]
a = [1.0, -8.047, 26.178, -50.390, 52.988, ...]
```

### Highpass 340 Hz (Order 7)
```
b = [0.92614234, -6.48299638, 16.20749097, ...]
a = [1.0, -5.78349584, 14.77825534, ...]
```

## FreeRTOS Tasks

### Task 1: WiFi & Audio (Core 0)
```
Name:        WiFi_Audio_Task
Priority:    1
Stack Size:  8 KB
Core:        0
```

### Task 2: ML Inference (Core 1)
```
Name:        ML_Inference_Task
Priority:    1
Stack Size:  16 KB
Core:        1
```

## Memory Layout
```
Audio Buffer 1:        96 KB (float[24000])
Audio Buffer 2:        96 KB (float[24000]) - for double buffering
Processed Audio:       96 KB (float[24000])
MFCC Features:         3.3 KB (float[13*64])
TFLite Tensor Arena:   200 KB (model tensors)
Code & Data:           ~50 KB
WiFi Stack:            ~30 KB
FreeRTOS:              ~20 KB
─────────────────────────────
Total (estimated):     ~350 KB (fits in ESP32 PSRAM)
```

## Inference Pipeline Timing
```
Audio Recording:       2000 ms
Preprocessing:         ~50 ms
MFCC Extraction:       ~80 ms
Inference:             ~100 ms
─────────────────────────────
Total Processing:      ~240 ms
```

## WiFi & Light Control
```
Port:        38899 (UDP)
Light MAC:   $(bedroom_light_mac) [from credentials.h]
Light IP:    Discovered automatically via getSystemConfig
```

## Class Mapping
```
0 = "ambiente"       (ambient/silence - no action)
1 = "apagarLuz"      (turn off light)
2 = "prenderLuz"     (turn on light)
```

## Confidence Threshold
```
Default: 0.6 (60%)
Can be adjusted in handlePrediction()
```

## Build Configuration (platformio.ini)
```
Platform:     espressif32
Board:        esp32doit-devkit-v1
Framework:    arduino
Monitor Baud: 115200
Build Flags:  -DTF_LITE_MICRO -O3
Libraries:    ArduinoJson, TensorFlow Lite Micro
```

## Serial Commands
```
'g' = Record and process audio
     • Records 2 seconds
     • Preprocesses
     • Extracts MFCC
     • Runs inference
     • Controls lights based on prediction
```

## Important Constants (header files)

### audio_processing.h
```cpp
#define SAMPLE_RATE 12000
#define RECORD_TIME 2
#define TOTAL_SAMPLES 24000
#define N_MFCC 13
#define MAX_TIME_STEPS 64
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f
```

### mfcc.h
```cpp
#define N_MFCC 13
#define N_FFT 512
#define HOP_LENGTH 160
#define N_MELS 40
#define F_MIN 0.0f
#define F_MAX 6000.0f
```

### tflite_inference.h
```cpp
#define INPUT_HEIGHT 13
#define INPUT_WIDTH 64
#define INPUT_CHANNELS 1
#define NUM_CLASSES 3
```

## GPIO Mapping
```
LED_BUILTIN:        Standard ESP32 LED
MICROPHONE_PIN:     34 (ADC input)
POTENTIOMETER_PIN:  35 (ADC input)
```

## Python Preprocessing Parameters (from training)

The C++ implementation must exactly match these parameters:

### Audio Processing (AudioProcessing.py)
```python
# Filters
bandpass_filter(data, 320.0, 3000.0, fs=12000, order=6)
highpass_filter(data, 340.0, fs=12000, order=7)

# Blur kernels
audio_blur(data, window_size=13)
gaussian_blur(data, window_size=15)
median_filter(data, kernel_size=11)

# Spectral subtraction
simple_spectral_subtraction(audio, noise_reduction_factor=1.3)

# Peak removal
kill_peaks(data, windows=5, min_data=0.6, threshold=60.0)
```

### MFCC Extraction (MFCCsDataSets.py)
```python
mfcc = librosa.feature.mfcc(y=y, sr=12000, n_mfcc=13)
# Padding: (13, 64)
# Normalization: (X - mean) / (std + 1e-8)
```

### Model Training (train_model.py)
```python
global_mean = -47.3197508271
global_std = 174.8682650662
X_scaled = (X - global_mean) / (global_std + 1e-8)
```

## Deployment Checklist

- [ ] TFLite model converted: `voiceModel_3classes.tflite`
- [ ] Model embedded in C header: `model_data.h` ✓
- [ ] Audio processing code complete: `audio_processing.cpp` ✓
- [ ] MFCC extraction code complete: `mfcc.cpp` ✓
- [ ] TFLite inference wrapper: `tflite_inference.cpp` ✓
- [ ] FreeRTOS main loop: `main.cpp` ✓
- [ ] platformio.ini configured ✓
- [ ] All headers created ✓
- [ ] Build passes without errors
- [ ] Upload to ESP32
- [ ] Test with voice commands

## Modifications for Production

1. **Remove debug Serial.println()** statements for speed
2. **Add SD card logging** if needed for diagnostics
3. **Implement error recovery** if inference fails
4. **Add low-power mode** for idle time
5. **Implement continuous recording** (optional, uses more power)
6. **Add LED feedback** for recording/processing state
7. **Increase confidence threshold** if too many false positives
8. **Retrain model** with more voice samples if accuracy is low

## Testing Commands

```bash
# Build only
pio run -e esp32doit-devkit-v1

# Build and upload
pio run -e esp32doit-devkit-v1 -t upload

# Clean and rebuild
pio run -e esp32doit-devkit-v1 -t clean
pio run -e esp32doit-devkit-v1

# Monitor serial output
pio device monitor -e esp32doit-devkit-v1 -b 115200

# Check free memory (add to code)
Serial.printf("Free PSRAM: %d KB\n", ESP.getFreePsram() / 1024);
Serial.printf("Free heap: %d KB\n", ESP.getFreeHeap() / 1024);
```
