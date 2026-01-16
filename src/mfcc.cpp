#include "mfcc.h"
#include <cmath>
#include <cstring>
#include <algorithm>

MFCCExtractor::MFCCExtractor() {
    fft_real.resize(N_FFT);
    fft_imag.resize(N_FFT);
    magnitude.resize(N_FFT / 2 + 1);
}

void MFCCExtractor::compute_fft(const float* input, int size) {
    // Aplicar ventana Hamming (como en Librosa)
    for (int i = 0; i < size; i++) {
        float win = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (size - 1));
        fft_real[i] = input[i] * win;
        fft_imag[i] = 0.0f;
    }

    // FFT Radix-2
    int n = size;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(fft_real[i], fft_real[j]);
            std::swap(fft_imag[i], fft_imag[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * M_PI / len;
        float wlen_r = cosf(ang), wlen_i = -sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f, w_i = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float u_r = fft_real[i+k], u_i = fft_imag[i+k];
                float v_r = fft_real[i+k+len/2]*w_r - fft_imag[i+k+len/2]*w_i;
                float v_i = fft_real[i+k+len/2]*w_i + fft_imag[i+k+len/2]*w_r;
                fft_real[i+k] = u_r + v_r; fft_imag[i+k] = u_i + v_i;
                fft_real[i+k+len/2] = u_r - v_r; fft_imag[i+k+len/2] = u_i - v_i;
                float next_w_r = w_r*wlen_r - w_i*wlen_i;
                w_i = w_r*wlen_i + w_i*wlen_r; w_r = next_w_r;
            }
        }
    }
    for (int i = 0; i <= size / 2; i++) {
        magnitude[i] = sqrtf(fft_real[i]*fft_real[i] + fft_imag[i]*fft_imag[i]);
    }
}

int MFCCExtractor::extract_mfcc(const float* audio, int audio_len, float* mfcc_output, int max_frames) {
    int num_frames = 0;
    for (int start = 0; start + N_FFT <= audio_len && num_frames < max_frames; start += HOP_LENGTH) {
        compute_fft(audio + start, N_FFT);
        // Aquí deberías aplicar tu banco de filtros Mel y DCT 
        // simplificado para obtener los 13 coeficientes.
        num_frames++;
    }
    return num_frames;
}

void MFCCExtractor::pad_to_fixed_length(float* mfcc, int num_frames, float* mfcc_padded, int target_frames) {
    for (int m = 0; m < N_MFCC; m++) {
        for (int t = 0; t < target_frames; t++) {
            if (t < num_frames) mfcc_padded[m * target_frames + t] = mfcc[m * num_frames + t];
            else mfcc_padded[m * target_frames + t] = 0.0f;
        }
    }
}