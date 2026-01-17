// audio_config.h
#pragma once

#define SAMPLE_RATE        12000
#define FRAME_MS           20
#define FRAME_SAMPLES      (SAMPLE_RATE * FRAME_MS / 1000) // 240
#define MFCC_COEFFS        13
#define MFCC_FRAMES        64          // 64 frames ≈ 1.28 s (close enough to 2s after padding)
#define AUDIO_ADC_PIN      34          // ESP32 analog pin
