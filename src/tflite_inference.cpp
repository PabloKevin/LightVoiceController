#include "tflite_inference.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"

// Global objects for TFLite
static tflite::MicroErrorReporter micro_error_reporter;
static const int kTensorArenaSize = 200 * 1024;  // 200KB for tensor arena
static uint8_t tensor_arena[kTensorArenaSize];

bool VoiceModelInference::initialize(const unsigned char* model_data) {
    // Load model schema
    const tflite::Model* model = tflite::GetModel(model_data);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // Version mismatch
        return false;
    }
    
    // Create resolver with operations needed for the model
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddDense();
    resolver.AddFlattenSimple();
    resolver.AddReshape();
    resolver.AddDropout();
    resolver.AddBatchToSpaceNd();
    resolver.AddSoftmax();
    resolver.AddAdd();
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        return false;
    }
    
    // Get input and output tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // Verify tensor shapes
    if (input_tensor->dims->size != 4) {
        return false;
    }
    
    if (output_tensor->dims->size != 2) {
        return false;
    }
    
    return true;
}

int VoiceModelInference::predict(float* mfcc) {
    // Copy MFCC to input buffer
    // MFCC format: (13 x 64)
    // Input tensor expects: (1, 13, 64, 1)
    
    if (input_tensor->type != kTfLiteFloat32) {
        return -1;  // Error
    }
    
    float* input_data = input_tensor->data.f;
    
    // Copy MFCC (13 x 64) into input (1 x 13 x 64 x 1)
    for (int m = 0; m < INPUT_HEIGHT; m++) {
        for (int t = 0; t < INPUT_WIDTH; t++) {
            int mfcc_idx = m * INPUT_WIDTH + t;
            int input_idx = m * INPUT_WIDTH + t;  // Batch dimension is 1
            input_data[input_idx] = mfcc[mfcc_idx];
        }
    }
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        return -1;  // Error
    }
    
    // Get output
    float* output_data = output_tensor->data.f;
    
    // Find class with maximum probability
    int best_class = 0;
    float best_prob = output_data[0];
    
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > best_prob) {
            best_prob = output_data[i];
            best_class = i;
        }
    }
    
    return best_class;
}

void VoiceModelInference::get_output_probabilities(float* probabilities) {
    if (output_tensor->type != kTfLiteFloat32) {
        return;
    }
    
    float* output_data = output_tensor->data.f;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        probabilities[i] = output_data[i];
    }
}

void VoiceModelInference::predict_with_confidence(int& class_id, float& confidence, float* mfcc) {
    // Run prediction
    class_id = predict(mfcc);
    
    // Get confidence score
    if (output_tensor->type == kTfLiteFloat32) {
        float* output_data = output_tensor->data.f;
        
        if (class_id >= 0 && class_id < NUM_CLASSES) {
            confidence = output_data[class_id];
        } else {
            confidence = 0.0f;
        }
    }
}

const char* VoiceModelInference::get_class_name(int class_id) {
    if (class_id >= 0 && class_id < NUM_CLASSES) {
        return CLASS_NAMES[class_id];
    }
    return "unknown";
}

void VoiceModelInference::reset() {
    // Reset interpreter state (clear tensor arena if needed)
    // For TFLite Micro, this typically isn't necessary between inferences
}

void VoiceModelInference::print_model_info() {
    // Print model information (for debugging)
    // Input shape
    if (input_tensor) {
        // Serial.print("Input shape: ");
        // for (int i = 0; i < input_tensor->dims->size; i++) {
        //     Serial.print(input_tensor->dims->data[i]);
        //     if (i < input_tensor->dims->size - 1) Serial.print(" x ");
        // }
        // Serial.println();
    }
    
    // Output shape
    if (output_tensor) {
        // Serial.print("Output shape: ");
        // for (int i = 0; i < output_tensor->dims->size; i++) {
        //     Serial.print(output_tensor->dims->data[i]);
        //     if (i < output_tensor->dims->size - 1) Serial.print(" x ");
        // }
        // Serial.println();
    }
}
