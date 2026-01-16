#include "tflite_inference.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ArduinoJson.h>

// Pre-trained model weights (quantized/simplified for embedded)
// These need to be extracted from your .keras model
// For now, using placeholder structure

VoiceModelInference::VoiceModelInference() : model_loaded(false) {
    // Allocate large buffers in heap (malloc uses PSRAM if available)
    conv1_out = (float*)malloc(6 * 64 * 16 * sizeof(float));
    pool1_out = (float*)malloc(6 * 64 * 16 * sizeof(float));
    conv2_out = (float*)malloc(6 * 32 * 16 * sizeof(float));
    pool2_out = (float*)malloc(3 * 32 * 16 * sizeof(float));
    conv3_out = (float*)malloc(3 * 32 * 32 * sizeof(float));
    flatten_out = (float*)malloc(3072 * sizeof(float));
    
    // Check allocation success
    if (!conv1_out || !pool1_out || !conv2_out || !pool2_out || !conv3_out || !flatten_out) {
        Serial.println("ERROR: Failed to allocate PSRAM buffers");
        model_loaded = false;
        return;
    }
    
    // Initialize small DRAM buffers
    std::fill(dense1_out, dense1_out + 64, 0.0f);
    std::fill(logits, logits + NUM_CLASSES, 0.0f);
    std::fill(output, output + NUM_CLASSES, 0.0f);
    
    model_loaded = true;
}

VoiceModelInference::~VoiceModelInference() {
    // Free heap buffers
    if (conv1_out) free(conv1_out);
    if (pool1_out) free(pool1_out);
    if (conv2_out) free(conv2_out);
    if (pool2_out) free(pool2_out);
    if (conv3_out) free(conv3_out);
    if (flatten_out) free(flatten_out);
}

bool VoiceModelInference::initialize() {
    // For this lightweight implementation, we load pre-trained weights
    // In production, extract weights from .keras model using Python script
    model_loaded = true;
    Serial.println("✓ Model initialized (lightweight inference engine)");
    return true;
}

void VoiceModelInference::batch_norm_2d(float* input, int h, int w, int c) {
    // BatchNormalization layer
    // Simplified: just apply per-channel normalization
    for (int ch = 0; ch < c; ch++) {
        float mean = 0.0f, var = 0.0f;
        int count = h * w;
        
        // Calculate mean
        for (int i = 0; i < h * w; i++) {
            mean += input[i * c + ch];
        }
        mean /= count;
        
        // Calculate variance
        for (int i = 0; i < h * w; i++) {
            float diff = input[i * c + ch] - mean;
            var += diff * diff;
        }
        var /= count;
        
        // Normalize
        float std = std::sqrt(var + 1e-5f);
        for (int i = 0; i < h * w; i++) {
            input[i * c + ch] = (input[i * c + ch] - mean) / std;
        }
    }
}

void VoiceModelInference::leaky_relu(float* data, int size, float alpha) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] *= alpha;
        }
    }
}

void VoiceModelInference::softmax_activation(float* data, int size) {
    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

void VoiceModelInference::conv2d_layer(float* input, int in_h, int in_w, int in_c,
                                       int out_c, int kernel_h, int kernel_w, int stride) {
    // Simplified 2D convolution (SAME padding)
    // This is a placeholder - actual weights would come from the model
    std::fill(conv1_out, conv1_out + in_h * in_w * out_c, 0.1f);
}

void VoiceModelInference::maxpool2d_layer(float* input, int in_h, int in_w, int in_c,
                                          float* output, int pool_h, int pool_w, int stride) {
    int out_h = (in_h + stride - 1) / stride;
    int out_w = (in_w + stride - 1) / stride;
    
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int c = 0; c < in_c; c++) {
                float max_val = -1e6f;
                
                for (int kh = 0; kh < pool_h; kh++) {
                    for (int kw = 0; kw < pool_w; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        if (ih < in_h && iw < in_w) {
                            float val = input[(ih * in_w + iw) * in_c + c];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                
                output[(oh * out_w + ow) * in_c + c] = max_val;
            }
        }
    }
}

void VoiceModelInference::flatten_layer(float* input, int h, int w, int c, float* output) {
    std::copy(input, input + h * w * c, output);
}

void VoiceModelInference::dense_layer(float* input, int input_size, int output_size,
                                      float* output, bool with_activation) {
    // Simplified dense layer (weights not implemented)
    // In production, weights would be loaded from model
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.1f;
    }
    
    if (with_activation) {
        leaky_relu(output, output_size, 0.2f);
    }
}

int VoiceModelInference::predict(float* mfcc) {
    if (!model_loaded) return -1;
    
    // Forward pass through network
    // [Input: 13×64] → BatchNorm → Conv2D → Pool → Conv2D → Pool → Conv2D → Flatten → Dense → Dense → Softmax
    
    // Batch Norm on input
    batch_norm_2d(mfcc, 13, 64, 1);
    
    // Conv2D (16 filters)
    // conv2d_layer(mfcc, 13, 64, 1, 16, 3, 3, 1);
    
    // For now, output placeholder predictions
    output[0] = 0.3f;
    output[1] = 0.3f;
    output[2] = 0.4f;
    
    softmax_activation(output, NUM_CLASSES);
    
    // Find class with max probability
    int best_class = 0;
    float best_prob = output[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output[i] > best_prob) {
            best_prob = output[i];
            best_class = i;
        }
    }
    
    return best_class;
}

void VoiceModelInference::get_output_probabilities(float* probabilities) {
    std::copy(output, output + NUM_CLASSES, probabilities);
}

void VoiceModelInference::predict_with_confidence(int& class_id, float& confidence, float* mfcc) {
    class_id = predict(mfcc);
    confidence = (class_id >= 0) ? output[class_id] : 0.0f;
}

const char* VoiceModelInference::get_class_name(int class_id) {
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return CLASS_NAMES[class_id];
    }
    return "unknown";
}