#include "audio_processing.h"
#include <cstring>
#include <algorithm>

AudioProcessor::AudioProcessor() {
    // Initialize filter states (10 is max order)
    bp_state.resize(10, 0.0f);
    hp_state.resize(10, 0.0f);
    
    // Create blur kernel for audio_blur (window_size=5)
    // kernel = np.ones(5) / 5
    blur_kernel.resize(5);
    for (int i = 0; i < 5; i++) {
        blur_kernel[i] = 1.0f / 5.0f;
    }
}

float AudioProcessor::apply_iir_filter(float sample, const float* b, const float* a,
                                       std::vector<float>& state, int order) {
    // Direct Form II implementation
    // y[n] = b[0]*x[n] + state[0]
    // state[0] = b[1]*x[n] - a[1]*y[n] + state[1]
    // ...
    
    float w = sample;
    for (int i = 1; i < order; i++) {
        w -= a[i] * state[i];
    }
    
    float output = b[0] * w;
    for (int i = 1; i < order; i++) {
        output += b[i] * state[i];
    }
    
    // Update state
    for (int i = order - 1; i > 0; i--) {
        state[i] = state[i - 1];
    }
    state[0] = w;
    
    return output;
}

void AudioProcessor::bandpass_filter(float* data, int len) {
    // Order 6 bandpass filter (320-3000 Hz)
    int order = 6;
    bp_state.assign(order, 0.0f);
    
    for (int i = 0; i < len; i++) {
        data[i] = apply_iir_filter(data[i], Filters::BP_B, Filters::BP_A, bp_state, order);
    }
}

void AudioProcessor::highpass_filter(float* data, int len) {
    // Order 7 highpass filter (340 Hz)
    int order = 7;
    hp_state.assign(order, 0.0f);
    
    for (int i = 0; i < len; i++) {
        data[i] = apply_iir_filter(data[i], Filters::HP_B, Filters::HP_A, hp_state, order);
    }
}

void AudioProcessor::audio_blur(float* data, int len, int window_size) {
    // Simple moving average: convolve with [1/5, 1/5, 1/5, 1/5, 1/5]
    float* temp = new float[len];
    std::copy(data, data + len, temp);
    
    int half_window = window_size / 2;
    for (int i = 0; i < len; i++) {
        float sum = 0.0f;
        int count = 0;
        
        for (int j = -half_window; j <= half_window; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < len) {
                sum += temp[idx];
                count++;
            }
        }
        data[i] = sum / count;
    }
    
    delete[] temp;
}

void AudioProcessor::gaussian_blur(float* data, int len, int window_size) {
    // Hamming window: w = hamming(window_size) / sum(hamming(window_size))
    float* temp = new float[len];
    std::copy(data, data + len, temp);
    
    // Create Hamming window
    float* hamming_win = new float[window_size];
    float sum_win = 0.0f;
    for (int i = 0; i < window_size; i++) {
        hamming_win[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (window_size - 1));
        sum_win += hamming_win[i];
    }
    
    // Normalize window
    for (int i = 0; i < window_size; i++) {
        hamming_win[i] /= sum_win;
    }
    
    // Apply convolution
    int half_window = window_size / 2;
    for (int i = 0; i < len; i++) {
        float result = 0.0f;
        for (int j = 0; j < window_size; j++) {
            int idx = i + j - half_window;
            if (idx >= 0 && idx < len) {
                result += temp[idx] * hamming_win[j];
            }
        }
        data[i] = result;
    }
    
    delete[] hamming_win;
    delete[] temp;
}

void AudioProcessor::median_filter(float* data, int len, int kernel_size) {
    // Simple median filter implementation
    float* temp = new float[len];
    std::copy(data, data + len, temp);
    
    int half_size = kernel_size / 2;
    for (int i = 0; i < len; i++) {
        float* window = new float[kernel_size];
        int count = 0;
        
        // Collect samples
        for (int j = -half_size; j <= half_size; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < len) {
                window[count++] = temp[idx];
            }
        }
        
        // Find median
        std::sort(window, window + count);
        data[i] = window[count / 2];
        
        delete[] window;
    }
    
    delete[] temp;
}

void AudioProcessor::remove_dc_offset(float* data, int len) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < len; i++) {
        mean += data[i];
    }
    mean /= len;
    
    // Subtract mean
    for (int i = 0; i < len; i++) {
        data[i] -= mean;
    }
}

void AudioProcessor::normalize_audio(float* data, int len) {
    // Find max absolute value
    float max_val = 0.0f;
    for (int i = 0; i < len; i++) {
        float abs_val = std::abs(data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // Normalize
    if (max_val > 1e-6f) {
        for (int i = 0; i < len; i++) {
            data[i] /= max_val;
        }
    }
}

void AudioProcessor::spectral_subtraction(float* data, int len, float noise_factor) {
    // Simplified spectral subtraction:
    // Estimate noise from first and last frames
    // This is a simplified version without full STFT
    
    int frame_size = len / 8;
    float noise_floor = 0.0f;
    
    // Estimate noise from first 4 frames
    for (int i = 0; i < frame_size * 4 && i < len; i++) {
        float power = data[i] * data[i];
        noise_floor += power;
    }
    noise_floor /= (frame_size * 4);
    
    // Subtract noise with spectral floor (0.01 * original_magnitude)
    for (int i = 0; i < len; i++) {
        float signal = std::abs(data[i]);
        float signal_power = signal * signal;
        
        float clean_power = signal_power - noise_factor * noise_floor;
        if (clean_power < 0.01f * signal_power) {
            clean_power = 0.01f * signal_power;
        }
        
        // Preserve sign
        if (data[i] >= 0) {
            data[i] = std::sqrt(clean_power);
        } else {
            data[i] = -std::sqrt(clean_power);
        }
    }
}

void AudioProcessor::kill_peaks(float* data, int len, int windows, float min_data, float threshold) {
    // Algorithm: divide audio into windows and remove frames with high energy spikes
    int samples_per_window = len / windows;
    
    // We'll use a simple approach: calculate energy per frame
    // and threshold based on local mean energy
    
    int trial = 0;
    float current_threshold = threshold;
    int max_trials = 10;
    
    float* result = new float[len];
    
    while (trial < max_trials) {
        int result_idx = 0;
        
        // Process each window
        for (int w = 0; w < windows; w++) {
            int start = w * samples_per_window;
            int end = (w == windows - 1) ? len : (w + 1) * samples_per_window;
            
            // Calculate frame energy
            float frame_energy = 0.0f;
            float frame_mean_energy = 0.0f;
            int valid_samples = 0;
            
            // First, calculate mean energy
            for (int i = start; i < end; i++) {
                float power = data[i] * data[i];
                frame_energy += power;
            }
            frame_mean_energy = frame_energy / (end - start);
            
            // Check if this frame should be kept
            if (frame_energy <= current_threshold * frame_mean_energy) {
                // Keep frame
                for (int i = start; i < end; i++) {
                    result[result_idx++] = data[i];
                }
            }
        }
        
        // Check if we have enough data
        if (result_idx >= len * min_data) {
            // Success - copy result back and pad if needed
            std::copy(result, result + result_idx, data);
            if (result_idx < len) {
                std::fill(data + result_idx, data + len, 0.0f);
            }
            break;
        } else if (trial >= max_trials - 1) {
            // Give up, keep original
            break;
        }
        
        // Lower threshold and try again
        current_threshold *= 0.7f;
        trial++;
    }
    
    delete[] result;
}

void AudioProcessor::process_complete_pipeline(float* raw_audio, int len, float* processed_audio) {
    // Create working buffer
    float* buffer = new float[len];
    std::copy(raw_audio, raw_audio + len, buffer);
    
    // Python pipeline order (from process_audio_wav):
    // 1. Normalize to [-1, 1]
    normalize_audio(buffer, len);
    
    // 2. Remove DC offset
    remove_dc_offset(buffer, len);
    
    // 3. Skip first 1/15 of data
    int skip_samples = len / 15;
    int working_len = len - skip_samples;
    std::copy(buffer + skip_samples, buffer + len, buffer);
    
    // 4. Apply blur filters
    audio_blur(buffer, working_len, 13);
    gaussian_blur(buffer, working_len, 15);
    median_filter(buffer, working_len, 11);
    
    // 5. Bandpass filter (320-3000 Hz)
    bandpass_filter(buffer, working_len);
    
    // 6. Remove initial and final samples from filter
    int trim_start = working_len / 32;
    int trim_end = working_len / 1024;
    working_len = working_len - trim_start - trim_end;
    std::copy(buffer + trim_start, buffer + trim_start + working_len, buffer);
    
    // 7. Highpass filter (340 Hz)
    highpass_filter(buffer, working_len);
    
    // 8. Kill peaks
    kill_peaks(buffer, working_len, 5, 0.6f, 60.0f);
    
    // 9. Spectral subtraction
    spectral_subtraction(buffer, working_len, 1.3f);
    
    // Copy to output
    std::copy(buffer, buffer + working_len, processed_audio);
    
    // Pad with zeros if needed
    if (working_len < len) {
        std::fill(processed_audio + working_len, processed_audio + len, 0.0f);
    }
    
    delete[] buffer;
}
