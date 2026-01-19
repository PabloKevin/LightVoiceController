#include "audio_preProcessing.h"
#include <algorithm>
#include <cstring>

void AudioProcessor::normalize_audio(audio* data, int len) {
    float max_val = 0.0001f;
    for (int i = 0; i < len; i++) {
        float val = std::abs(data->processed[i]);
        if (val > max_val) max_val = val;
    }
    for (int i = 0; i < len; i++) {
        float norm_val = data->processed[i] / max_val;
        data->processed[i] = norm_val;
    }
}

void AudioProcessor::remove_dc_offset(audio* data, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) mean += data->processed[i];
    mean /= len;
    for (int i = 0; i < len; i++) data->processed[i] -= mean;
}

// Optimización: Usamos una ventana pequeña en lugar de duplicar todo el audio
void AudioProcessor::audio_blur(audio* data, int len, int window_size) {
    float sum = 0;
    // Buffer circular pequeño para no duplicar los 96KB
    for (int i = 0; i < len; i++) {
        sum = 0;
        int count = 0;
        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {
            if (j >= 0 && j < len) {
                sum += data->processed[j];
                count++;
            }
        }
        data->processed[i] = sum / count;
    }
}

// Optimización: Usamos un array fijo en el stack para la ventana del filtro de mediana
void AudioProcessor::median_filter(audio* data, int len, int kernel_size) {
    // Usamos un buffer pequeño en el STACK (memoria local rápida)
    float window[21]; // kernel_size máximo de 21
    int k_half = kernel_size / 2;

    for (int i = k_half; i < len - k_half; i++) {
        for(int j = 0; j < kernel_size; j++) {
            window[j] = data->processed[i - k_half + j];
        }
        std::sort(window, window + kernel_size);
        data->processed[i] = window[k_half];
    }
}

void AudioProcessor::kill_peaks(audio* data, int len, float threshold) {
    for (int i = 0; i < len; i++) {
        if (std::abs(data->processed[i]) > threshold) data->processed[i] = (data->processed[i] > 0 ? threshold : -threshold) * 0.1f;
    }
}

void AudioProcessor::process_complete_pipeline(audio* data, int len) {
    if (data == nullptr) return;

    // 1. Quitar offset y normalizar base
    remove_dc_offset(data, len);
    normalize_audio(data, len);
    
    // 2. Skip inicial (reducimos el puntero, no copiamos)
    int skip = len / 15;
    int working_len = len - skip;
    // Movemos los datos hacia adelante para sobreescribir el 'skip'
    std::memmove(data, data + skip, working_len * sizeof(float));

    // 3. Filtros (Ahora no consumen RAM extra)
    median_filter(data, working_len, 11);
    audio_blur(data, working_len, 13);
    
    // 4. Kill Peaks
    kill_peaks(data, working_len, 0.6f);
    
    // 5. Normalización final para que el MFCC sea consistente
    normalize_audio(data, working_len);
}