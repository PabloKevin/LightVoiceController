#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include <cstdint>
#include <cmath>
#include <cstring>

// --- Model Configuration ---
#define INPUT_HEIGHT 13      // MFCC coefficients
#define INPUT_WIDTH 64       // Time steps
#define NUM_CLASSES 3        // ambiente, apagarLuz, prenderLuz

// Class indices
#define CLASS_AMBIENTE 0
#define CLASS_APAGAR_LUZ 1
#define CLASS_PRENDER_LUZ 2

// Class names (static to avoid multiple definition)
static constexpr const char* CLASS_NAMES[] = {
    "ambiente",
    "apagarLuz",
    "prenderLuz"
};

/**
 * Lightweight Neural Network Inference Engine for ESP32
 * Executes the pre-trained voice model without external dependencies
 */
class VoiceModelInference {
private:
    // Model state
    bool model_loaded;
    
    // Large intermediate buffers (use PSRAM to save DRAM)
    float* conv1_out;    // Conv2D output (6 * 64 * 16 floats)
    float* pool1_out;    // MaxPool output
    float* conv2_out;    // Conv2D output
    float* pool2_out;    // MaxPool output
    float* conv3_out;    // Conv2D output
    float* flatten_out;  // Flatten output
    
    // Small buffers in DRAM (frequently accessed)
    float dense1_out[64];            // Dense output
    float logits[NUM_CLASSES];       // Final logits
    float output[NUM_CLASSES];       // Softmax probabilities
    
    // Layer execution functions
    void batch_norm_2d(float* input, int h, int w, int c);
    void conv2d_layer(float* input, int in_h, int in_w, int in_c,
                     int out_c, int kernel_h, int kernel_w, int stride);
    void maxpool2d_layer(float* input, int in_h, int in_w, int in_c,
                        float* output, int pool_h, int pool_w, int stride);
    void flatten_layer(float* input, int h, int w, int c, float* output);
    void dense_layer(float* input, int input_size, int output_size,
                    float* output, bool with_activation);
    void leaky_relu(float* data, int size, float alpha = 0.2f);
    void softmax_activation(float* data, int size);
    
public:
    VoiceModelInference();
    ~VoiceModelInference();
    
    /**
     * Initialize the model (loads pre-trained weights)
     */
    bool initialize();
    
    /**
     * Run inference on MFCC input
     * @param mfcc: Input MFCC features (13 x 64)
     * @return Prediction class (0, 1, or 2)
     */
    int predict(float* mfcc);
    
    /**
     * Get output probabilities
     */
    void get_output_probabilities(float* probabilities);
    
    /**
     * Get prediction with confidence
     */
    void predict_with_confidence(int& class_id, float& confidence, float* mfcc);
    
    /**
     * Get class name string
     */
    const char* get_class_name(int class_id);
    
    /**
     * Check if model is ready
     */
    bool is_ready() { return model_loaded; }
};

#endif // TFLITE_INFERENCE_H