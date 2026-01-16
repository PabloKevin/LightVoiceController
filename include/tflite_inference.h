#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// --- Model Configuration ---
#define INPUT_HEIGHT 13      // MFCC coefficients
#define INPUT_WIDTH 64       // Time steps
#define INPUT_CHANNELS 1     // Grayscale
#define NUM_CLASSES 3        // ambiente, apagarLuz, prenderLuz

// Class indices
#define CLASS_AMBIENTE 0
#define CLASS_APAGAR_LUZ 1
#define CLASS_PRENDER_LUZ 2

// Class names
const char* CLASS_NAMES[] = {
    "ambiente",
    "apagarLuz",
    "prenderLuz"
};

/**
 * TFLite Inference Engine for Voice Recognition
 * Runs the voiceModel_3classes.tflite model on ESP32
 */
class VoiceModelInference {
private:
    // TFLite components
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    
    // Input buffer: (1, 13, 64, 1)
    float input_buffer[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    
public:
    /**
     * Initialize the inference engine with the model data
     * @param model_data: Pointer to the .tflite model data (in PROGMEM on ESP32)
     * @param error_reporter: Custom error reporter (optional)
     * @return true if initialization successful
     */
    bool initialize(const unsigned char* model_data);
    
    /**
     * Run inference on MFCC input
     * @param mfcc: Input MFCC features (13 x 64)
     * @return Prediction result (0, 1, or 2)
     */
    int predict(float* mfcc);
    
    /**
     * Get raw output probabilities
     * @param probabilities: Output array (must be size >= NUM_CLASSES)
     */
    void get_output_probabilities(float* probabilities);
    
    /**
     * Get prediction with confidence score
     * @param class_id: Output class ID (0, 1, or 2)
     * @param confidence: Output confidence (0.0 to 1.0)
     */
    void predict_with_confidence(int& class_id, float& confidence, float* mfcc);
    
    /**
     * Get class name from prediction
     */
    const char* get_class_name(int class_id);
    
    /**
     * Reset interpreter state (if needed)
     */
    void reset();
    
    /**
     * Get model info
     */
    void print_model_info();
};

#endif // TFLITE_INFERENCE_H
