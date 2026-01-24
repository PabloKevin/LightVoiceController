#include "mfcc.h"
#include <cmath>

MFCCExtractor::MFCCExtractor() {
    // Reservar memoria una sola vez para evitar fragmentación
    window = new float[N_FFT];
    fft_buffer = new float[N_FFT * 2];

    // Inicializar DSP del ESP32
    dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    
    // Pre-calcular ventana de Hann
    dsps_wind_hann_f32(window, N_FFT);
}

MFCCExtractor::~MFCCExtractor() {
    delete[] window;
    delete[] fft_buffer;
}

int MFCCExtractor::extract_mfcc(const float* audio, int audio_len, float* mfcc_output) {
    int current_window = 0;

    // Limpiar el buffer de salida inicialmente (por si el audio es corto)
    memset(mfcc_output, 0, sizeof(float) * N_MFCC * TARGET_WINDOWS);

    for (int start = 0; start + N_FFT <= audio_len && current_window < TARGET_WINDOWS; start += HOP_LENGTH) {
        
        // 1. Preparar Frame (Windowing)
        for (int i = 0; i < N_FFT; i++) {
            fft_buffer[i * 2] = audio[start + i] * window[i];
            fft_buffer[i * 2 + 1] = 0;
        }

        // 2. FFT (Acelerada por Hardware)
        dsps_fft2r_fc32(fft_buffer, N_FFT);
        dsps_bit_rev_fc32(fft_buffer, N_FFT);

        // 3. Magnitud y Banco de Filtros Mel (OPTIMIZADO)
        float log_mel[N_MFCC];
        for (int m = 0; m < N_MFCC; m++) {
            float energy = 0;
            
            // Solo recorremos los bins donde el filtro m tiene valores > 0
            uint16_t s_bin = pgm_read_word(&mel_start_bin[m]);
            uint16_t e_bin = pgm_read_word(&mel_end_bin[m]);

            for (int i = s_bin; i <= e_bin; i++) {
                float real = fft_buffer[i * 2];
                float imag = fft_buffer[i * 2 + 1];
                float mag = sqrtf(real * real + imag * imag);
                energy += mag * pgm_read_float(&mel_weights[m][i]);
            }
            log_mel[m] = logf(energy + 1e-6f);
        }

        // 4. DCT-II con normalización Ortho
        for (int i = 0; i < N_MFCC; i++) {
            float sum = 0;
            for (int j = 0; j < N_MFCC; j++) {
                sum += log_mel[j] * cosf(M_PI * i * (j + 0.5f) / N_MFCC);
            }
            
            float scale = (i == 0) ? sqrtf(1.0f / N_MFCC) : sqrtf(2.0f / N_MFCC);
            float final_val = sum * scale;

            // Guardar en formato (N_MFCC x 64) -> Row Major
            // mfcc_output[m * 64 + t]
            mfcc_output[i * TARGET_WINDOWS + current_window] = final_val;
        }
        current_window++;
    }

    return current_window; // Retorna cuántas ventanas se llenaron realmente
}