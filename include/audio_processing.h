#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#include <cmath>
#include <vector>

// Definiciones globales de audio
#define SAMPLE_RATE 12000
#define RECORD_TIME 2
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_TIME)
#define N_MFCC 13
#define MAX_TIME_STEPS 64

// Parámetros de normalización (de tu entrenamiento)
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f

class AudioProcessor {
public:
    AudioProcessor() {}
    void normalize_audio(float* data, int len);
    void remove_dc_offset(float* data, int len);
    void audio_blur(float* data, int len, int window_size);
    void median_filter(float* data, int len, int kernel_size);
    void kill_peaks(float* data, int len, float threshold);
    void process_complete_pipeline(float* audio, int len);
};

#endif