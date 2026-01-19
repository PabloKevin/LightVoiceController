#pragma once

#include <cmath>
#include <vector>
#include "audio_recording.h"

// Parámetros de normalización (de tu entrenamiento)
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f

class AudioProcessor {
public:
    AudioProcessor() {}
    void normalize_audio(audio* data, int len);
    void remove_dc_offset(audio* data, int len);
    void audio_blur(audio* data, int len, int window_size);
    void median_filter(audio* data, int len, int kernel_size);
    void kill_peaks(audio* data, int len, float threshold);
    void process_complete_pipeline(audio* audio, int len);
};

