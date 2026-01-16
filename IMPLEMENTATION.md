# ESP32 Voice Recognition System - Implementation Guide

## Overview

This project implements a complete voice recognition system on the ESP32 with dual-core FreeRTOS task architecture:

- **Core 0**: Handles WiFi communication and audio recording
- **Core 1**: Performs ML inference (MFCC extraction + TensorFlow Lite model)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ESP32 Dual-Core System                     │
├──────────────────────┬──────────────────────────────────┤
│   Core 0 (WiFi)      │   Core 1 (ML Inference)          │
├──────────────────────┼──────────────────────────────────┤
│ • Record Audio       │ • Extract MFCC                   │
│ • WiFi Comms         │ • Normalize Features             │
│ • Light Control      │ • Run TFLite Model               │
│ • Potentiometer      │ • Handle Predictions             │
└──────────────────────┴──────────────────────────────────┘
         ↓ (FreeRTOS Queue)
   Audio Ready Signal
```

## Components

### 1. Audio Processing (`audio_processing.h/cpp`)

Replicates the Python preprocessing pipeline exactly:

```
Raw Audio
   ↓
Normalize to [-1, 1]
   ↓
Remove DC offset
   ↓
Skip first 1/15 of samples
   ↓
Audio Blur (5-point moving average)
   ↓
Gaussian Blur (15-point Hamming window)
   ↓
Median Filter (11-point)
   ↓
Bandpass Filter (320-3000 Hz, Butterworth order 6)
   ↓
Highpass Filter (340 Hz, Butterworth order 7)
   ↓
Kill Peaks (remove high-energy frames)
   ↓
Spectral Subtraction (noise reduction)
   ↓
Processed Audio
```

**Key functions:**
- `bandpass_filter()` - IIR Butterworth bandpass
- `highpass_filter()` - IIR Butterworth highpass
- `audio_blur()` - Simple convolution
- `gaussian_blur()` - Hamming window convolution
- `median_filter()` - Non-linear filtering
- `kill_peaks()` - Remove outliers
- `spectral_subtraction()` - Noise reduction
- `process_complete_pipeline()` - Full preprocessing

### 2. MFCC Extraction (`mfcc.h/cpp`)

Implements Mel-Frequency Cepstral Coefficients:

```
Processed Audio
   ↓
Apply Hann Window
   ↓
Radix-2 FFT (512-point)
   ↓
Magnitude Spectrum
   ↓
Mel Filterbank (40 mel-bands)
   ↓
Log Transform (dB scale)
   ↓
DCT-II (Discrete Cosine Transform)
   ↓
Keep first 13 coefficients
   ↓
Pad/Truncate to 64 frames
   ↓
Normalize MFCC (global mean/std)
   ↓
Model Input: (1, 13, 64, 1)
```

**Configuration:**
- N_FFT: 512
- HOP_LENGTH: 160 samples (~13ms at 12kHz)
- N_MELS: 40 mel bands
- N_MFCC: 13 coefficients
- MAX_TIME_STEPS: 64 frames

### 3. TensorFlow Lite Inference (`tflite_inference.h/cpp`)

Runs the embedded Keras model:

```
TFLite Model (voiceModel_3classes.tflite)
├─ Input: (1, 13, 64, 1) float
├─ Layers: Conv2D → MaxPool → Dense → Softmax
├─ Output: (1, 3) - Class probabilities
└─ Size: 204 KB
```

**Classes:**
- 0: `ambiente` (ambient/silence)
- 1: `apagarLuz` (turn off light)
- 2: `prenderLuz` (turn on light)

## File Structure

```
LightVoiceController/
├── include/
│   ├── audio_processing.h      # Audio filtering & preprocessing
│   ├── mfcc.h                  # MFCC feature extraction
│   ├── tflite_inference.h      # TFLite model inference
│   ├── model_data.h            # Embedded .tflite model (AUTO-GENERATED)
│   └── credentials.h           # WiFi & device MAC
├── src/
│   ├── main.cpp                # FreeRTOS dual-core main
│   ├── audio_processing.cpp    # Audio filtering implementation
│   ├── mfcc.cpp                # MFCC extraction implementation
│   └── tflite_inference.cpp    # Model inference implementation
├── DSP/
│   ├── MachineLearning/
│   │   ├── model_weights/
│   │   │   ├── best_models/
│   │   │   │   └── voiceModel_FullTrain_3classes_best.keras
│   │   │   └── tflite_models/
│   │   │       └── voiceModel_3classes.tflite
│   │   └── convert_to_tflite.py
│   └── DataSets/
│       ├── AudioProcessing.py
│       └── MFCCsDataSets.py
├── platformio.ini              # PlatformIO configuration
└── convert_tflite_to_header.py # Model → C header converter
```

## Setup and Build

### Prerequisites

1. **PlatformIO** (VSCode extension or CLI)
2. **Python 3.8+** with TensorFlow (for model conversion)
3. **ESP32 board** (ESP32-DevKit-V1 or compatible)

### Build Steps

#### 1. Convert Keras Model to TFLite

```bash
conda activate Control  # or your ML environment
cd DSP/MachineLearning
python3 convert_to_tflite.py
```

This creates: `model_weights/tflite_models/voiceModel_3classes.tflite`

#### 2. Convert TFLite to C Header

```bash
cd /path/to/LightVoiceController
python3 convert_tflite_to_header.py
```

This creates: `include/model_data.h` (204 KB model embedded in code)

#### 3. Build with PlatformIO

```bash
# Using PlatformIO CLI
pio run -e esp32doit-devkit-v1

# Or in VSCode: Click "Build" in PlatformIO sidebar
```

#### 4. Upload to ESP32

```bash
pio run -e esp32doit-devkit-v1 -t upload

# Or: Click "Upload" in PlatformIO sidebar
```

#### 5. Monitor Serial Output

```bash
pio device monitor -e esp32doit-devkit-v1 -b 115200

# Or: Click "Serial Monitor" in PlatformIO sidebar
```

## Usage

### Voice Commands

Send `g` via serial monitor to record and process audio:

1. **Record**: Records 2 seconds of audio at 12 kHz (24,000 samples)
2. **Preprocess**: Applies complete audio filtering pipeline
3. **MFCC**: Extracts 13×64 feature matrix
4. **Inference**: Runs TFLite model
5. **Control**: Executes corresponding light action

### Output

```
=== Voice Recognition System Starting ===
Initializing TFLite model...
✓ Model initialized successfully

=== FreeRTOS Tasks Created ===
Core 0: WiFi + Audio Recording
Core 1: ML Inference

Send 'g' via serial to record audio and run inference

[User sends 'g']
>>> Starting audio recording (2 sec)...
>>> Recording finished. Time: 2000 ms

=== Processing Audio ===
1. Preprocessing audio...
2. Extracting MFCC features...
   Extracted 61 frames
3. Padding to fixed length...
4. Normalizing MFCC...
5. Running inference...

Processing time: 245 ms
Prediction: Class=2, Confidence=95.23%
Class Name: prenderLuz
Turning on light...
Sent: Brightness 100%, Temp 4100K
```

## Memory Usage

| Component | Size |
|-----------|------|
| Audio Buffer (2s @ 12kHz) | 96 KB (2× float) |
| MFCC Buffer (13×64) | 3.3 KB |
| Tensor Arena (TFLite) | 200 KB |
| Code/Data | ~50 KB |
| **Total** | **~350 KB** (fits in ESP32 PSRAM) |

## Performance

| Operation | Time |
|-----------|------|
| Audio Recording | ~2000 ms |
| Preprocessing | ~50 ms |
| MFCC Extraction | ~80 ms |
| Inference | ~100 ms |
| **Total Pipeline** | ~240 ms |

## Important Notes

### Critical: Python → C++ Preprocessing Match

The preprocessing pipeline **must exactly match** the Python implementation:

✓ Same filter coefficients (Butterworth bandpass/highpass)
✓ Same frame sizes and padding
✓ Same MFCC parameters (13 coeffs, 64 frames, 12 kHz)
✓ Same normalization (global mean: -47.32, std: 174.87)

If preprocessing differs, model accuracy will **degrade significantly**.

### IIR Filter Implementation

Butterworth filter coefficients are pre-computed using:
```python
from scipy.signal import butter
b, a = butter(order, [low, high], btype='band')
```

They're stored as constants in `audio_processing.h`.

### FFT Implementation

Simple radix-2 FFT suitable for ESP32:
- 512-point FFT
- Direct Cooley-Tukey algorithm
- No optimizations (use if needed for speed)

### DCT Implementation

Slow DCT-II for accuracy:
```
Y[k] = sqrt(1/N) * sum if k=0, else sqrt(2/N) * sum
```

Can be optimized with FFT-based DCT if processing time is critical.

## Troubleshooting

### Model Inference Fails

1. Check model file is embedded: `ls -l include/model_data.h`
2. Verify model initialization: Check serial output for "✓ Model initialized"
3. Check tensor arena size in `tflite_inference.cpp` (200 KB may not be enough)

### Audio Preprocessing Issues

1. Verify microphone input: Check raw ADC values in serial
2. Test filters individually: Uncomment debug code in `audio_processing.cpp`
3. Compare output with Python version: Export audio from ESP32

### Low Accuracy

1. **Preprocessing**: Ensure filters match Python exactly
2. **MFCC normalization**: Verify mean/std values (-47.32, 174.87)
3. **Audio quality**: Ensure microphone is not saturating
4. **Confidence threshold**: Adjust in `handlePrediction()` if needed

## Future Improvements

1. **Optimize FFT**: Use esp-dsp library for faster FFT
2. **Quantize Model**: Convert TFLite to INT8 for smaller size
3. **Multi-class**: Train model with all 10 commands
4. **Online Learning**: Adapt model to user's voice
5. **Wake Word**: Add "hey ESP" detection before recording

## References

- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
- [ESP32 FreeRTOS](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/freertos.html)
- [MFCC Algorithm](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [Butterworth Filter](https://en.wikipedia.org/wiki/Butterworth_filter)

## License

MIT
