#include "audio_preProcessing.h"
int working_len;

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

void AudioProcessor::gaussian_blur(float* data, int len) {
    int kernel_size = 15;
    float kernel[kernel_size] = {0.0104712, 0.01643381, 0.03314067, 0.05728277, 0.08407849, 0.10822059
                                , 0.12492744, 0.13089005, 0.12492744, 0.10822059, 0.08407849, 0.05728277
                                , 0.03314067, 0.01643381, 0.0104712};

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


const float HighPassFilter::b[8] = { 0.66985341f, -4.68897389f, 14.06692168f, -23.44486948f, 23.44486948f, -14.06692168f, 4.68897389f, -0.66985341f };
const float HighPassFilter::a[8] = { 1.0f, -6.20011746f, 16.51609997f, -24.49999193f, 21.85468509f, -11.72180355f, 3.49983534f, -0.44870359f };

float HighPassFilter::process(float x) {
    // Ecuación de diferencias (Direct Form II)
    float current_w = x - a[1]*w[1] - a[2]*w[2] - a[3]*w[3] - a[4]*w[4] - a[5]*w[5] - a[6]*w[6] - a[7]*w[7];
    
    float y = b[0]*current_w + b[1]*w[1] + b[2]*w[2] + b[3]*w[3] + b[4]*w[4] + b[5]*w[5] + b[6]*w[6] + b[7]*w[7];

    // Desplazar estados (mantenimiento del buffer)
    for(int i = 7; i > 1; i--) {
        w[i] = w[i-1];
    }
    w[1] = current_w;

    return y;
}

void HighPassFilter::processArray(float* data, int len) {
    for(int i = 0; i < len; i++) {
        data[i] = process(data[i]);
    }
}

void HighPassFilter::reset() {
    for(int i = 0; i < 8; i++) {
        w[i] = 0.0f;
    }
}


const float BandPassFilter::b[13] = { 0.017385727403559214, 0.0, -0.10431436442135528, 0.0, 0.2607859110533882, 0.0, -0.3477145480711843, 0.0, 0.2607859110533882, 0.0, -0.10431436442135528, 0.0, 0.017385727403559214 };
const float BandPassFilter::a[13] = { 1.0, -5.605388986184154, 14.267909254449279, -22.56210489068331, 25.701855963278657, -22.66033567899688, 15.62648218865623, -8.325849995057787, 3.4285001999730036, -1.0804031979900046, 0.23854575710321907, -0.03174939252338663, 0.0025810369570774478 }; 

float BandPassFilter::process(float x) {
    // Ecuación de diferencias (Direct Form II)
    float current_w = x - a[1]*w[1] - a[2]*w[2] - a[3]*w[3] - a[4]*w[4] - a[5]*w[5] - a[6]*w[6] - a[7]*w[7];
    
    float y = b[0]*current_w + b[1]*w[1] + b[2]*w[2] + b[3]*w[3] + b[4]*w[4] + b[5]*w[5] + b[6]*w[6] + b[7]*w[7];

    // Desplazar estados (mantenimiento del buffer)
    for(int i = 13; i > 1; i--) {
        w[i] = w[i-1];
    }
    w[1] = current_w;

    return y;
}

void BandPassFilter::processArray(float* data, int len) {
    for(int i = 0; i < len; i++) {
        data[i] = process(data[i]);
    }
}

void BandPassFilter::reset() {
    for(int i = 0; i < 13; i++) {
        w[i] = 0.0f;
    }
}


HighPassFilter highPass;
BandPassFilter bandPass;



void AudioProcessor::process_complete_pipeline(float* window_pointer, int len) {
    unsigned long startTime = millis();
    Serial.println("Comenzando preprocesamiento...");
    int windows = 1;
    int window_len = TOTAL_SAMPLES/windows;

    for (int i=0; i<windows; i++){
        // Copiar datos de la ventana original
        window_pointer = window_pointer + (i*window_len);

        // 1. Quitar offset y normalizar base
        remove_dc_offset(window_pointer, window_len);
        normalize_audio(window_pointer, window_len);
        
        
        // 2. Skip inicial (reducimos el puntero, no copiamos)
        int skip = len / 15;
        window_pointer += skip;
        working_len = len - skip;
        // Movemos los datos hacia adelante para sobreescribir el 'skip'
        //std::memmove(window_pointer, data + skip, window_len * sizeof(float));
        

        // 3. Filtros (Ahora no consumen RAM extra)
        audio_blur(window_pointer, working_len, 13);
        gaussian_blur(window_pointer, working_len);
        median_filter(window_pointer, working_len, 11);
        
        apply_bandpass(window_pointer, working_len, SAMPLE_RATE, 320.0f, 3000.0f);
        skip = len / 32;
        window_pointer += skip;
        working_len = len - skip - len/1024;

        highPass.processArray(window_pointer, working_len);
        //normalize_audio(window_pointer, window_len);
    }
    Serial.printf(">>> Preprocesamiento finalizado. Tiempo total: %lu ms\n", millis() - startTime);
}