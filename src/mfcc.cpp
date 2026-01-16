#include "mfcc.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// Fast Walsh-Hadamard DCT approximation for ESP32
void dct_type2_slow(float* data, int N) {
    // Slow but accurate DCT-II implementation
    // Used for converting mel-log-spectrogram to MFCC
    float* temp = new float[N];
    
    for (int k = 0; k < N; k++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += data[n] * std::cos(M_PI * k * (n + 0.5f) / N);
        }
        temp[k] = sum;
    }
    
    // Normalization
    temp[0] *= std::sqrt(1.0f / N);
    for (int k = 1; k < N; k++) {
        temp[k] *= std::sqrt(2.0f / N);
    }
    
    std::copy(temp, temp + N, data);
    delete[] temp;
}

MFCCExtractor::MFCCExtractor() {
    // Pre-allocate buffers
    fft_real.resize(N_FFT);
    fft_imag.resize(N_FFT);
    magnitude.resize(N_FFT / 2 + 1);
    mel_spectrogram.resize(N_MELS);
    
    create_mel_filterbank();
    create_dct_matrix();
}

float MFCCExtractor::hz_to_mel(float hz) {
    // Convert Hz to Mel scale
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MFCCExtractor::mel_to_hz(float mel) {
    // Convert Mel to Hz
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void MFCCExtractor::create_mel_filterbank() {
    // Create mel-scaled triangular filterbank
    // Similar to librosa.filters.mel()
    
    int num_fft_bins = N_FFT / 2 + 1;
    mel_filterbank.resize(N_MELS);
    
    // Convert frequency limits to mel
    float mel_min = hz_to_mel(F_MIN);
    float mel_max = hz_to_mel(F_MAX);
    
    // Create linearly spaced mel centers
    std::vector<float> mel_centers(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; i++) {
        mel_centers[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);
    }
    
    // Create triangular filters
    for (int m = 0; m < N_MELS; m++) {
        mel_filterbank[m].resize(num_fft_bins, 0.0f);
        
        float left_mel = mel_centers[m];
        float center_mel = mel_centers[m + 1];
        float right_mel = mel_centers[m + 2];
        
        float left_hz = mel_to_hz(left_mel);
        float center_hz = mel_to_hz(center_mel);
        float right_hz = mel_to_hz(right_mel);
        
        // Convert Hz to FFT bin index
        float left_bin = left_hz * N_FFT / 12000.0f;  // SAMPLE_RATE = 12000
        float center_bin = center_hz * N_FFT / 12000.0f;
        float right_bin = right_hz * N_FFT / 12000.0f;
        
        // Create triangular filter
        for (int k = 0; k < num_fft_bins; k++) {
            float bin = (float)k;
            float val = 0.0f;
            
            if (bin >= left_bin && bin <= center_bin) {
                val = (bin - left_bin) / (center_bin - left_bin);
            } else if (bin > center_bin && bin <= right_bin) {
                val = (right_bin - bin) / (right_bin - center_bin);
            }
            
            mel_filterbank[m][k] = val;
        }
    }
}

void MFCCExtractor::create_dct_matrix() {
    // DCT is applied in extract_mfcc using dct_type2_slow
    // This function is here for future optimization
}

void MFCCExtractor::apply_window(float* signal, int size) {
    // Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
    for (int n = 0; n < size; n++) {
        float window = 0.5f * (1.0f - std::cos(2.0f * M_PI * n / (size - 1)));
        signal[n] *= window;
    }
}

void MFCCExtractor::radix2_fft(float* real, float* imag, int size) {
    // Simple radix-2 FFT implementation (Cooley-Tukey)
    // For ESP32, this is sufficient for N_FFT=512
    
    // Bit reversal
    for (int i = 0; i < size; i++) {
        int j = 0;
        int tmp_i = i;
        for (int k = 1; k < size; k *= 2) {
            j = (j << 1) | (tmp_i & 1);
            tmp_i >>= 1;
        }
        
        if (j > i) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    
    // Butterflies
    for (int s = 1; s <= (int)std::log2(size); s++) {
        int m = 1 << s;
        float wr = std::cos(-2.0f * M_PI / m);
        float wi = std::sin(-2.0f * M_PI / m);
        
        for (int k = 0; k < size; k += m) {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            
            for (int j = 0; j < m / 2; j++) {
                for (int i = j; i < size; i += m) {
                    int t = i + m / 2;
                    
                    // Complex multiply: (w_real + w_imag*j) * (real[t] + imag[t]*j)
                    float tr = w_real * real[t] - w_imag * imag[t];
                    float ti = w_real * imag[t] + w_imag * real[t];
                    
                    real[t] = real[i] - tr;
                    imag[t] = imag[i] - ti;
                    real[i] = real[i] + tr;
                    imag[i] = imag[i] + ti;
                }
                
                // Update twiddle factor
                float wr_tmp = w_real * wr - w_imag * wi;
                w_imag = w_real * wi + w_imag * wr;
                w_real = wr_tmp;
            }
        }
    }
}

void MFCCExtractor::mel_scale(float* magnitude, int fft_bins, float* mel_output) {
    // Apply mel filterbank to magnitude spectrum
    // mel_output = mel_filterbank @ magnitude
    
    for (int m = 0; m < N_MELS; m++) {
        float sum = 0.0f;
        for (int k = 0; k < fft_bins; k++) {
            sum += mel_filterbank[m][k] * magnitude[k];
        }
        // Avoid log(0)
        mel_output[m] = std::max(sum, 1e-10f);
    }
}

int MFCCExtractor::extract_mfcc(const float* audio, int audio_len,
                                float* mfcc_output, int max_frames) {
    // Calculate number of frames
    int num_frames = (audio_len - N_FFT) / HOP_LENGTH + 1;
    if (num_frames > max_frames) {
        num_frames = max_frames;
    }
    
    int fft_bins = N_FFT / 2 + 1;
    
    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        int start_sample = frame_idx * HOP_LENGTH;
        
        // Extract frame
        std::fill(fft_real.begin(), fft_real.end(), 0.0f);
        std::fill(fft_imag.begin(), fft_imag.end(), 0.0f);
        
        for (int i = 0; i < N_FFT && start_sample + i < audio_len; i++) {
            fft_real[i] = audio[start_sample + i];
        }
        
        // Apply Hann window
        apply_window(fft_real.data(), N_FFT);
        
        // Compute FFT
        radix2_fft(fft_real.data(), fft_imag.data(), N_FFT);
        
        // Compute magnitude spectrum
        for (int k = 0; k < fft_bins; k++) {
            float real_part = fft_real[k];
            float imag_part = fft_imag[k];
            magnitude[k] = std::sqrt(real_part * real_part + imag_part * imag_part);
        }
        
        // Apply mel filterbank
        mel_scale(magnitude.data(), fft_bins, mel_spectrogram.data());
        
        // Convert to dB scale: 20 * log10(mel_spectrogram)
        float* mel_log = new float[N_MELS];
        for (int m = 0; m < N_MELS; m++) {
            mel_log[m] = 20.0f * std::log10(mel_spectrogram[m]);
        }
        
        // Apply DCT to get MFCCs
        dct_type2_slow(mel_log, N_MELS);
        
        // Keep only first N_MFCC coefficients
        for (int m = 0; m < N_MFCC; m++) {
            mfcc_output[m * num_frames + frame_idx] = mel_log[m];
        }
        
        delete[] mel_log;
    }
    
    return num_frames;
}

void MFCCExtractor::pad_to_fixed_length(float* mfcc, int num_frames,
                                        float* mfcc_padded, int target_frames) {
    // MFCC format: (N_MFCC, num_frames)
    // Pad or truncate to (N_MFCC, target_frames)
    
    if (num_frames <= target_frames) {
        // Pad with zeros
        for (int m = 0; m < N_MFCC; m++) {
            // Copy existing frames
            for (int t = 0; t < num_frames; t++) {
                mfcc_padded[m * target_frames + t] = mfcc[m * num_frames + t];
            }
            // Zero-pad remaining frames
            for (int t = num_frames; t < target_frames; t++) {
                mfcc_padded[m * target_frames + t] = 0.0f;
            }
        }
    } else {
        // Truncate
        for (int m = 0; m < N_MFCC; m++) {
            for (int t = 0; t < target_frames; t++) {
                mfcc_padded[m * target_frames + t] = mfcc[m * num_frames + t];
            }
        }
    }
}

/*
void MFCCExtractor::normalize_mfcc(float* mfcc, int num_coeffs, int num_frames) {
    // Normalize using global mean and std
    // (X - MFCC_MEAN) / (MFCC_STD + 1e-8)
    // Note: This matches training normalization
    
    for (int m = 0; m < num_coeffs; m++) {
        for (int t = 0; t < num_frames; t++) {
            int idx = m * num_frames + t;
            mfcc[idx] = (mfcc[idx] - MFCC_MEAN) / (MFCC_STD + 1e-8f);
        }
    }
}
*/