#include "audio_processing.h"
#include <algorithm>
#include <cstring>

void AudioProcessor::normalize_audio(float* data, int len) {
    float max_val = 0.0001f;
    for (int i = 0; i < len; i++) {
        if (std::abs(data[i]) > max_val) max_val = std::abs(data[i]);
    }
    for (int i = 0; i < len; i++) data[i] /= max_val;
}

void AudioProcessor::remove_dc_offset(float* data, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) mean += data[i];
    mean /= len;
    for (int i = 0; i < len; i++) data[i] -= mean;
}

void AudioProcessor::audio_blur(float* data, int len, int window_size) {
    std::vector<float> temp(data, data + len);
    for (int i = 0; i < len; i++) {
        float sum = 0;
        int count = 0;
        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {
            if (j >= 0 && j < len) {
                sum += temp[j];
                count++;
            }
        }
        data[i] = sum / count;
    }
}

void AudioProcessor::median_filter(float* data, int len, int kernel_size) {
    std::vector<float> temp(data, data + len);
    std::vector<float> window(kernel_size);
    for (int i = kernel_size / 2; i < len - kernel_size / 2; i++) {
        for(int j=0; j<kernel_size; j++) window[j] = temp[i - kernel_size/2 + j];
        std::sort(window.begin(), window.end());
        data[i] = window[kernel_size / 2];
    }
}

void AudioProcessor::kill_peaks(float* data, int len, float threshold) {
    for (int i = 0; i < len; i++) {
        if (std::abs(data[i]) > threshold) data[i] *= 0.1f;
    }
}

void AudioProcessor::process_complete_pipeline(float* audio, int len) {
    normalize_audio(audio, len);
    remove_dc_offset(audio, len);
    
    // Skip inicial (len/15) como en Python
    int skip = len / 15;
    int working_len = len - skip;
    std::memmove(audio, audio + skip, working_len * sizeof(float));

    // Filtros
    median_filter(audio, working_len, 11);
    audio_blur(audio, working_len, 13);
    
    // Kill Peaks (min_data=0.6)
    kill_peaks(audio, working_len, 0.6f);
    
    // Normalización final
    normalize_audio(audio, working_len);
}