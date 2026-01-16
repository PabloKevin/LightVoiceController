#ifndef TFLITE_INFERENCE_H
#define TFLITE_INFERENCE_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define NUM_CLASSES 3
#define CLASS_AMBIENTE 0
#define CLASS_APAGAR_LUZ 1
#define CLASS_PRENDER_LUZ 2

// Declaración externa
extern const char* CLASS_NAMES[];

class VoiceModelInference {
private:
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;

public:
    bool initialize(const unsigned char* model_data);
    int predict(float* mfcc);
    void predict_with_confidence(int& class_id, float& confidence, float* mfcc);
    const char* get_class_name(int class_id);
    void get_output_probabilities(float* probabilities);
};

#endif