#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#include <cmath>
#include <cstring>
#include <vector>

// --- Audio Processing Configuration ---
#define SAMPLE_RATE 12000
#define RECORD_TIME 2
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_TIME)
#define N_MFCC 13
#define MAX_TIME_STEPS 64

// --- Normalization Parameters (from training data) ---
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f

// --- Butterworth Filter Coefficients ---
// Pre-computed coefficients for different filter types at FS=12000Hz

struct ButterworthCoeffs {
    float b[5];  // Numerator coefficients
    float a[5];  // Denominator coefficients (a[0] = 1.0)
};

// Butterworth Bandpass Filter: 320Hz - 3000Hz, order=6
// Computed offline using scipy.signal.butter
namespace Filters {
    // Bandpass 320-3000 Hz, order 6
    constexpr float BP_B[] = {
        0.00000033f, 0.00000000f, -0.00000330f, 0.00000000f, 0.00000825f,
        0.00000660f, -0.00003289f, 0.00006578f, -0.00003289f, 0.00000660f
    };
    constexpr float BP_A[] = {
        1.00000000f, -8.04693816f, 26.17814046f, -50.39008713f, 52.98825896f,
        -34.82752991f, 13.70815277f, -3.41850305f, 0.51999084f, -0.03254783f
    };
    
    // Highpass 340 Hz, order 7
    constexpr float HP_B[] = {
        0.92614234f, -6.48299638f, 16.20749097f, -21.61332129f, 16.20749097f,
        -6.48299638f, 0.92614234f
    };
    constexpr float HP_A[] = {
        1.00000000f, -5.78349584f, 14.77825534f, -18.90090234f, 13.95055999f,
        -5.96381765f, 1.16509313f
    };
}

class AudioProcessor {
private:
    // IIR Filter state
    std::vector<float> bp_state;  // Bandpass filter state
    std::vector<float> hp_state;  // Highpass filter state
    
    // Kernel for audio blur (convolution)
    std::vector<float> blur_kernel;
    
public:
    AudioProcessor();
    
    // --- Core preprocessing functions (matching Python) ---
    
    /**
     * Apply IIR Butterworth bandpass filter (320-3000 Hz)
     * Equivalent to: bandpass_filter(data, 320.0, 3000.0, fs=12000, order=6)
     */
    void bandpass_filter(float* data, int len);
    
    /**
     * Apply IIR Butterworth highpass filter (340 Hz)
     * Equivalent to: highpass_filter(data, 340.0, fs=12000, order=7)
     */
    void highpass_filter(float* data, int len);
    
    /**
     * Apply convolution-based blur
     * Equivalent to: audio_blur(data, window_size=5)
     */
    void audio_blur(float* data, int len, int window_size = 5);
    
    /**
     * Apply median filtering
     * Equivalent to: medfilt(data, kernel_size=11)
     */
    void median_filter(float* data, int len, int kernel_size = 11);
    
    /**
     * Apply Gaussian blur using Hamming window
     * Equivalent to: gaussian_blur(data, window_size=15)
     */
    void gaussian_blur(float* data, int len, int window_size = 15);
    
    /**
     * Remove DC offset
     * Equivalent to: data = data - np.mean(data)
     */
    void remove_dc_offset(float* data, int len);
    
    /**
     * Normalize audio to [-1, 1] range
     */
    void normalize_audio(float* data, int len);
    
    /**
     * Apply spectral subtraction for noise reduction
     * Simplified version (not full STFT, but frame-based estimation)
     * Equivalent to: simple_spectral_subtraction(audio, noise_reduction_factor=1.3)
     */
    void spectral_subtraction(float* data, int len, float noise_factor = 1.3f);
    
    /**
     * Kill peaks algorithm - removes high-energy frames
     * Equivalent to: kill_peaks(data, windows=5, min_data=0.6, threshold=60.0)
     */
    void kill_peaks(float* data, int len, int windows = 5, float min_data = 0.6f, float threshold = 60.0f);
    
    /**
     * Complete preprocessing pipeline
     * Replicates: process_audio_wav() from Python
     */
    void process_complete_pipeline(float* raw_audio, int len, float* processed_audio);
    
private:
    // Helper functions
    float apply_iir_filter(float sample, const float* b, const float* a, 
                          std::vector<float>& state, int order);
};

#endif // AUDIO_PROCESSING_H
