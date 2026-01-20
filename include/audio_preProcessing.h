#pragma once

#include <cmath>
#include <vector>
#include "audio_recording.h"
//#include "dsps_convolution.h"
#include "esp_dsp.h"
#include <algorithm>
#include <cstring>

// Parámetros de normalización (de tu entrenamiento)
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f

class AudioProcessor {
public:
    AudioProcessor() {}
    void normalize_audio(float* data, int len);
    void remove_dc_offset(float* data, int len);
    void audio_blur(float* data, int len, int kernel_size);
    void median_filter(float* data, int len, int kernel_size);
    void process_complete_pipeline(float* audio, int len);
    void apply_bandpass(float* data, int len, float fs, float lf_cut, float hf_cut);
};

void convolve_1d_same(float* input, int input_len, const float* kernel, int kernel_len, float* output);