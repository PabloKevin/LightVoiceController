#include "audio_processing.h"
#include <algorithm>
#include <cstring>

void AudioProcessor::normalize_audio(float* data, int len) {
    float max_val = 0.0001f;
    for (int i = 0; i < len; i++) {
        float val = std::abs(data[i]);
        if (val > max_val) max_val = val;
    }
    for (int i = 0; i < len; i++) data[i] /= max_val;
}

void AudioProcessor::remove_dc_offset(float* data, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) mean += data[i];
    mean /= len;
    for (int i = 0; i < len; i++) data[i] -= mean;
}

// Optimización: Usamos una ventana pequeña en lugar de duplicar todo el audio
void AudioProcessor::audio_blur(float* data, int len, int window_size) {
    float sum = 0;
    // Buffer circular pequeño para no duplicar los 96KB
    for (int i = 0; i < len; i++) {
        sum = 0;
        int count = 0;
        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {
            if (j >= 0 && j < len) {
                sum += data[j];
                count++;
            }
        }
        data[i] = sum / count;
    }
}

// Optimización: Usamos un array fijo en el stack para la ventana del filtro de mediana
void AudioProcessor::median_filter(float* data, int len, int kernel_size) {
    // Usamos un buffer pequeño en el STACK (memoria local rápida)
    float window[21]; // kernel_size máximo de 21
    int k_half = kernel_size / 2;

    for (int i = k_half; i < len - k_half; i++) {
        for(int j = 0; j < kernel_size; j++) {
            window[j] = data[i - k_half + j];
        }
        std::sort(window, window + kernel_size);
        data[i] = window[k_half];
    }
}

void AudioProcessor::kill_peaks(float* data, int len, float threshold) {
    for (int i = 0; i < len; i++) {
        if (std::abs(data[i]) > threshold) data[i] = (data[i] > 0 ? threshold : -threshold) * 0.1f;
    }
}

void AudioProcessor::process_complete_pipeline(float* audio, int len) {
    if (audio == nullptr) return;

    // 1. Quitar offset y normalizar base
    remove_dc_offset(audio, len);
    normalize_audio(audio, len);
    
    // 2. Skip inicial (reducimos el puntero, no copiamos)
    int skip = len / 15;
    int working_len = len - skip;
    // Movemos los datos hacia adelante para sobreescribir el 'skip'
    std::memmove(audio, audio + skip, working_len * sizeof(float));

    // 3. Filtros (Ahora no consumen RAM extra)
    median_filter(audio, working_len, 11);
    audio_blur(audio, working_len, 13);
    
    // 4. Kill Peaks
    kill_peaks(audio, working_len, 0.6f);
    
    // 5. Normalización final para que el MFCC sea consistente
    normalize_audio(audio, working_len);
}