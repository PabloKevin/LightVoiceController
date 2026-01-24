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

float* AudioProcessor::kill_peaks(float* audio, int len, int windows, float min_data, float threshold, int trials) {
    int samplesW = len / windows;
    int trial = 0;
    float current_threshold = threshold;

    // Buffer para marcar qué ventanas sobreviven (1 = vive, 0 = silenciada)
    bool* mask = new bool[windows];
    
    while (trial <= trials) {
        int kept_samples = 0;

        for (int w = 0; w < windows; w++) {
            int start_idx = w * samplesW;
            float max_sq = 0;
            float sum_sq = 0;

            // Calcular potencia máxima y promedio de la ventana
            for (int i = 0; i < samplesW; i++) {
                float val_sq = audio[start_idx + i] * audio[start_idx + i];
                if (val_sq > max_sq) max_sq = val_sq;
                sum_sq += val_sq;
            }

            float mean_sq = sum_sq / samplesW;

            // Lógica de detección de picos (Evitar división por cero)
            if (mean_sq > 0 && max_sq > (current_threshold * mean_sq)) {
                mask[w] = false; // Marcar para borrar
            } else {
                mask[w] = true;  // Marcar para mantener
                kept_samples += samplesW;
            }
        }

        // Condición de salida: ¿Mantuvimos suficiente audio?
        if (kept_samples >= len * min_data) {
            break;
        } else {
            current_threshold *= 0.7f; // Relajar el umbral
            trial++;
        }
    }

    // Si fallan todos los intentos, devolvemos el audio original (no se borra nada)
    if (trial > trials) {
        Serial.println("KillPeaks: Máximos intentos alcanzados, devolviendo audio original.");
        working_len = len;
        delete[] mask;
        return audio;
    }

    // Aplicar la máscara y reconstruir el audio (In-place para ahorrar RAM)
    // Nota: A diferencia de Python que concatena, aquí silenciamos o desplazamos.
    // Para mantener la lógica de "concatenate", vamos a sobreescribir el buffer original.
    
    int write_idx = 0;
    for (int w = 0; w < windows; w++) {
        if (mask[w]) {
            // Si la ventana se mantiene y no es la posición actual, la desplazamos
            if (write_idx != w * samplesW) {
                memmove(&audio[write_idx], &audio[w * samplesW], samplesW * sizeof(float));
            }
            write_idx += samplesW;
        }
    }

    // Actualizamos la longitud global de datos útiles
    working_len = write_idx;

    delete[] mask;
    
    return audio; 
}

#include "esp_dsp.h" // Librería oficial de procesamiento de señales del ESP32

void AudioProcessor::simple_spectral_subtraction(float* audio, int len, float noise_reduction_factor) {
    const int fft_size = 512; // Tamaño estándar para STFT
    const int hop_size = 256; // 50% de solapamiento
    
    // Inicializar DSP del ESP32
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (ret  != ESP_OK) return;

    // Buffers temporales
    float* window = new float[fft_size];
    float* noise_estimation = new float[fft_size / 2 + 1];
    float* fft_buffer = new float[fft_size * 2]; // Complejo (R, I, R, I...)
    
    // 1. Crear ventana de Hann
    dsps_wind_hann_f32(window, fft_size);
    for(int i=0; i < (fft_size/2+1); i++) noise_estimation[i] = 0;

    // 2. ESTIMACIÓN DE RUIDO (Primeros 4 frames)
    int noise_frames = 4;
    for (int f = 0; f < noise_frames; f++) {
        // Preparar buffer con ventana
        for (int i = 0; i < fft_size; i++) {
            fft_buffer[i * 2] = audio[f * hop_size + i] * window[i];
            fft_buffer[i * 2 + 1] = 0;
        }
        dsps_fft2r_fc32(fft_buffer, fft_size);
        
        // Acumular magnitud
        for (int i = 0; i <= fft_size / 2; i++) {
            float mag = sqrtf(fft_buffer[i*2]*fft_buffer[i*2] + fft_buffer[i*2+1]*fft_buffer[i*2+1]);
            noise_estimation[i] += mag / noise_frames;
        }
    }

    // 3. PROCESAMIENTO POR VENTANAS (Limpieza)
    // Para simplificar y ahorrar RAM, lo haremos "in-place" sobre el audio original
    // Nota: El iSTFT es complejo, aquí aplicamos la reducción de ganancia espectral.
    
    for (int step = 0; step < len - fft_size; step += hop_size) {
        // FFT del frame actual
        for (int i = 0; i < fft_size; i++) {
            fft_buffer[i * 2] = audio[step + i] * window[i];
            fft_buffer[i * 2 + 1] = 0;
        }
        dsps_fft2r_fc32(fft_buffer, fft_size);

        // Sustracción espectral
        for (int i = 0; i <= fft_size / 2; i++) {
            float real = fft_buffer[i * 2];
            float imag = fft_buffer[i * 2 + 1];
            float mag = sqrtf(real * real + imag * imag);
            
            // Restar ruido con "Spectral Floor" del 1%
            float new_mag = mag - (noise_reduction_factor * noise_estimation[i]);
            if (new_mag < 0.01f * mag) new_mag = 0.01f * mag;
            
            // Aplicar nueva magnitud manteniendo la fase original
            float gain = new_mag / (mag + 1e-9f);
            fft_buffer[i * 2] *= gain;
            fft_buffer[i * 2 + 1] *= gain;
            
            // Simetría para la IFFT
            if (i > 0 && i < fft_size / 2) {
                fft_buffer[(fft_size - i) * 2] = fft_buffer[i * 2];
                fft_buffer[(fft_size - i) * 2 + 1] = -fft_buffer[i * 2 + 1];
            }
        }

        // 4. IFFT (Reconstrucción)
        // Si dsps_ifft2r_fc32 no aparece, usa la lógica de FFT inversa estándar:
        // 1. Conjugamos el complejo (negamos la parte imaginaria)
        for (int i = 0; i < fft_size * 2; i += 2) fft_buffer[i + 1] = -fft_buffer[i + 1];

        // 2. Ejecutamos FFT normal
        dsps_fft2r_fc32(fft_buffer, fft_size);

        // 3. Conjugamos de nuevo y dividimos por N (tamaño de FFT)
        for (int i = 0; i < fft_size * 2; i++) {
            fft_buffer[i] = fft_buffer[i] / fft_size;
        }
        
        // Overlap-add (Suma y solapa)
        for (int i = 0; i < fft_size; i++) {
            // Re-aplicar ventana para suavizar la unión
            audio[step + i] = fft_buffer[i * 2] * window[i]; 
        }
    }

    // Limpieza
    delete[] window;
    delete[] noise_estimation;
    delete[] fft_buffer;
}

const float HighPassFilter::b[8] = { 0.66985341f, -4.68897389f, 14.06692168f, -23.44486948f, 23.44486948f, -14.06692168f, 4.68897389f, -0.66985341f };
const float HighPassFilter::a[8] = { 1.0f, -6.20011746f, 16.51609997f, -24.49999193f, 21.85468509f, -11.72180355f, 3.49983534f, -0.44870359f };

HighPassFilter::HighPassFilter() { 
    for(int i=0; i<13; i++) w[i] = 0; 
}

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


const float BandPassFilter::sos[6][6] = {
    {0.017385727403559214f, 0.03477145480711843f, 0.017385727403559214f, 1.0f, -0.18557006598195191f, 0.03571352726661195f},
    {1.0f, 2.0f, 1.0f, 1.0f, -0.08981556530109971f, 0.22427706792438623f},
    {1.0f, 2.0f, 1.0f, 1.0f, -0.013560968893866595f, 0.6390164437993713f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.6489274830445289f, 0.682675646210554f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.7663891777109502f, 0.7954820774559893f},
    {1.0f, -2.0f, 1.0f, 1.0f, -1.9011257252517564f, 0.9285809384371537f},
};

BandPassFilter::BandPassFilter() { 
    for(int i=0; i<6; i++) { w[i][0] = 0.0f; w[i][1] = 0.0f; }
}

float BandPassFilter::process(float x) {
    float out = x;
    for (int i = 0; i < 6; i++) {
        // Cada sección es un filtro Biquad estándar
        float b0 = sos[i][0], b1 = sos[i][1], b2 = sos[i][2];
        float a1 = sos[i][4], a2 = sos[i][5]; // sos[i][3] siempre es 1.0

        float current_w = out - a1 * w[i][0] - a2 * w[i][1];
        out = b0 * current_w + b1 * w[i][0] + b2 * w[i][1];

        w[i][1] = w[i][0];
        w[i][0] = current_w;
    }
    return out;
}

void BandPassFilter::processArray(float* data, int len) {
    for(int i = 0; i < len; i++) {
        data[i] = process(data[i]);
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
        working_len = len - skip;
        std::memmove(window_pointer, window_pointer + skip, working_len * sizeof(float));
        // Movemos los datos hacia adelante para sobreescribir el 'skip'
        
        

        // 3. Filtros (Ahora no consumen RAM extra)
        audio_blur(window_pointer, working_len, 13);
        gaussian_blur(window_pointer, working_len);
        median_filter(window_pointer, working_len, 11);
        
        bandPass.processArray(window_pointer, working_len);
        skip = len / 32;
        working_len = len - skip - len/1024;
        std::memmove(window_pointer, window_pointer + skip, working_len * sizeof(float));

        highPass.processArray(window_pointer, working_len);

        kill_peaks(window_pointer, working_len, 5, 0.6);
        //simple_spectral_subtraction(window_pointer, working_len);
        normalize_audio(window_pointer, working_len);
    }
    Serial.printf(">>> Preprocesamiento finalizado. Tiempo total: %lu ms\n", millis() - startTime);
}