#include "tflite_inference.h"
#include <cstring>

// Definición de nombres (solo aquí)
const char* CLASS_NAMES[] = {"ambiente", "apagarLuz", "prenderLuz"};

// 80KB es suficiente para la mayoría de modelos de audio en ESP32
static const int kTensorArenaSize =40 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

bool VoiceModelInference::initialize(const unsigned char* model_data) {
    const tflite::Model* model = tflite::GetModel(model_data);
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddAdd();
    resolver.AddLeakyRelu();
    resolver.AddMul();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) return false;

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    return true;
}

int VoiceModelInference::predict(float* mfcc) {
    std::memcpy(input_tensor->data.f, mfcc, 13 * 64 * sizeof(float));
    if (interpreter->Invoke() != kTfLiteOk) return -1;
    
    float* output_data = output_tensor->data.f;
    int max_idx = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_data[i] > output_data[max_idx]) max_idx = i;
    }
    return max_idx;
}

void VoiceModelInference::predict_with_confidence(int& class_id, float& confidence, float* mfcc) {
    class_id = predict(mfcc);
    if (class_id >= 0) confidence = output_tensor->data.f[class_id];
    else confidence = 0.0f;
}

const char* VoiceModelInference::get_class_name(int class_id) {
    if (class_id >= 0 && class_id < NUM_CLASSES) return CLASS_NAMES[class_id];
    return "unknown";
}

void VoiceModelInference::get_output_probabilities(float* probabilities) {
    std::memcpy(probabilities, output_tensor->data.f, NUM_CLASSES * sizeof(float));
}