#include "audio_preProcessing.h"


void convolve_1d_same(float* input, int input_len, const float* kernel, int kernel_len, float* output) {
    int pad = kernel_len / 2;
    
    // Usamos un buffer temporal para no contaminar los datos de entrada durante el proceso
    float* temp_out = new float[input_len];

    for (int i = 0; i < input_len; i++) {
        float sum = 0;
        for (int j = 0; j < kernel_len; j++) {
            int input_idx = i + j - pad;
            
            // Manejo de bordes (Zero-padding)
            if (input_idx >= 0 && input_idx < input_len) {
                sum += input[input_idx] * kernel[j];
            }
        }
        temp_out[i] = sum;
    }
    
    // Copiar resultado final y liberar memoria
    memcpy(output, temp_out, sizeof(float) * input_len);
    delete[] temp_out;
}


class BandPassFilter {
    private:
        // Coeficientes del filtro
        float b0, b1, b2, a1, a2;
        // Estados del filtro (memoria)
        float z1, z2;

    public:
        BandPassFilter(float lowcut, float highcut, float fs) {
            // Cálculo de coeficientes para un filtro Biquad Paso Banda
            float wo = 2.0f * M_PI * ((lowcut + highcut) / 2.0f) / fs;
            float bw = 2.0f * M_PI * (highcut - lowcut) / fs;
            float alpha = sin(wo) * sinh(log(2.0f) / 2.0f * bw * wo / sin(wo));

            float a0 = 1.0f + alpha;
            b0 = alpha / a0;
            b1 = 0.0f;
            b2 = -alpha / a0;
            a1 = (-2.0f * cos(wo)) / a0;
            a2 = (1.0f - alpha) / a0;

            z1 = z2 = 0.0f;
        }

        float process(float input) {
            float output = b0 * input + z1;
            z1 = b1 * input - a1 * output + z2;
            z2 = b2 * input - a2 * output;
            return output;
        }
};


void AudioProcessor::normalize_audio(float* data, int len) {
    float max_val = 0.0001f;
    for (int i = 0; i < len; i++) {
        float val = std::abs(data[i]);
        if (val > max_val) max_val = val;
    }
    for (int i = 0; i < len; i++) {
        float norm_val = data[i] / max_val;
        data[i] = norm_val;
    }
}

void AudioProcessor::remove_dc_offset(float* data, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) mean += data[i];
    mean /= len;
    for (int i = 0; i < len; i++) data[i] -= mean;
}

// Optimización: Usamos una ventana pequeña en lugar de duplicar todo el audio
void AudioProcessor::audio_blur(float* data, int len, int kernel_size) {
    if (kernel_size>128) kernel_size=128;
    float kernel[kernel_size];

    for (int i=0; i<kernel_size; i++){
        kernel[i] = 1.0f/kernel_size;
    }
    convolve_1d_same(data, len, kernel, kernel_size, data);
}

void AudioProcessor::median_filter(float* data, int len, int kernel_size) {
    if (kernel_size > 21) kernel_size = 21;
    if (kernel_size < 3) return; // Un filtro de mediana de 1 no hace nada

    // Necesitamos un buffer temporal para no leer datos ya filtrados
    float* temp_output = new float[len];
    // Copiamos los bordes que no se filtrarán
    std::memcpy(temp_output, data, len * sizeof(float));

    float window[kernel_size]; // Buffer estático en el stack para la ventana
    int k_half = kernel_size / 2;

    for (int i = k_half; i < len - k_half; i++) {
        // Llenar la ventana con los datos ORIGINALES
        for(int j = 0; j < kernel_size; j++) {
            window[j] = data[i - k_half + j];
        }
        
        // Ordenar para encontrar la mediana
        std::sort(window, window + kernel_size);
        
        // Guardar en el buffer TEMPORAL
        temp_output[i] = window[k_half];
    }

    // Copiar el resultado final al buffer original
    std::memcpy(data, temp_output, len * sizeof(float));
    delete[] temp_output;
}

void AudioProcessor::apply_bandpass(float* data, int len, float fs, float lf_cut, float hf_cut) {
    // Definimos el rango para voz (típico de telefonía para ahorrar ruido)
    BandPassFilter filter(lf_cut, hf_cut, fs);

    for (int i = 0; i < len; i++) {
        data[i] = filter.process(data[i]);
    }
}

void AudioProcessor::process_complete_pipeline(float* data, int len) {
    int windows = 1;
    int window_len = TOTAL_SAMPLES/windows;

    for (int i=0; i<windows; i++){
        // Copiar datos de la ventana original
        float* window_pointer = data + (i*window_len);

        // 1. Quitar offset y normalizar base
        remove_dc_offset(window_pointer, window_len);
        normalize_audio(window_pointer, window_len);
        
        /*
        // 2. Skip inicial (reducimos el puntero, no copiamos)
        int skip = len / 15;
        int working_len = len - skip;
        // Movemos los datos hacia adelante para sobreescribir el 'skip'
        std::memmove(window_pointer, data + skip, window_len * sizeof(float));
        */

        // 3. Filtros (Ahora no consumen RAM extra)
        audio_blur(window_pointer, window_len, 13);
        median_filter(window_pointer, window_len, 11);
        
        apply_bandpass(window_pointer, window_len, SAMPLE_RATE, 320.0f, 3000.0f);
    }
}