#ifndef MFCC_H
#define MFCC_H

#include <vector>
#include <cmath>

// --- MFCC Configuration ---
#define N_MFCC 13           // Number of MFCC coefficients
#define N_FFT 512           // FFT size
#define HOP_LENGTH 160      // Samples between successive frames (12000/75 ≈ 160)
#define N_MELS 40           // Number of mel bands (typically 40 for speech)
#define F_MIN 0.0f          // Minimum frequency in Hz
#define F_MAX 6000.0f       // Maximum frequency in Hz (half of SAMPLE_RATE)

/**
 * MFCC (Mel-Frequency Cepstral Coefficients) Extractor
 * 
 * Matches librosa.feature.mfcc(y=y, sr=12000, n_mfcc=13)
 * Output shape: (13, num_frames) where num_frames ≈ 64 for 2-second audio
 */
class MFCCExtractor {
private:
    // Pre-computed mel filterbank and other matrices
    std::vector<std::vector<float>> mel_filterbank;
    std::vector<float> dct_matrix;  // Discrete Cosine Transform matrix
    
    // Temporary buffers
    std::vector<float> fft_real;
    std::vector<float> fft_imag;
    std::vector<float> magnitude;
    std::vector<float> mel_spectrogram;
    
    // FFT computation
    void compute_fft(const float* input, int size);
    void apply_window(float* signal, int size);
    void mel_scale(float* magnitude, int fft_bins, float* mel_output);
    
public:
    MFCCExtractor();
    
    /**
     * Extract MFCC features from audio
     * Input: raw audio (float array)
     * Output: MFCC matrix (N_MFCC x num_frames)
     * 
     * Returns the number of frames (time steps) extracted
     */
    int extract_mfcc(const float* audio, int audio_len, 
                     float* mfcc_output, int max_frames);
    
    /**
     * Pad or truncate MFCC to fixed size (64 frames)
     */
    void pad_to_fixed_length(float* mfcc, int num_frames, 
                            float* mfcc_padded, int target_frames);
    
    /**
     * Normalize MFCC using global mean and std
     * Equivalent to: (X - MFCC_MEAN) / (MFCC_STD + 1e-8)
     */
    void normalize_mfcc(float* mfcc, int num_coeffs, int num_frames);
    
private:
    // Helper functions
    float hz_to_mel(float hz);
    float mel_to_hz(float mel);
    void create_mel_filterbank();
    void create_dct_matrix();
    void radix2_fft(float* real, float* imag, int size);
};

#endif // MFCC_H
