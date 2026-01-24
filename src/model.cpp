#include "model.h"

// Definición de constantes
const float mean = -50.281225f;
const float std_ = 159.534246f;

// Ya no definimos el array aquí, solo el puntero
uint8_t* tensor_arena = nullptr; 
const int kTensorArenaSize = 70 * 1024; // Un poco de margen

// Punteros globales
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

bool setup_MLmodel() {
    // 1. Pedir memoria dinámicamente
    if (tensor_arena == nullptr) {
        // Usamos MALLOC_CAP_INTERNAL para asegurar que sea RAM rápida
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (tensor_arena == nullptr) {
            Serial.println("¡Error! No hay RAM para el Tensor Arena");
            return false;
        }
    }

    tflite::InitializeTarget();
    model = tflite::GetModel(voiceModel_3classes_tflite);

    static tflite::AllOpsResolver op_resolver;
    static tflite::MicroErrorReporter micro_error_reporter;

    // El intérprete también debe ser dinámico para poder borrarlo
    if (interpreter != nullptr) delete interpreter;
    
    interpreter = new tflite::MicroInterpreter(
        model, op_resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error en AllocateTensors");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    return true;
}

// Nueva función para liberar TODO
void free_MLmodel() {
    if (interpreter != nullptr) {
        delete interpreter;
        interpreter = nullptr;
    }
    if (tensor_arena != nullptr) {
        heap_caps_free(tensor_arena);
        tensor_arena = nullptr;
    }
    Serial.println("Memoria de ML liberada.");
}


int predict(float* inputBuff_pointer){
    // 1. Quantize and Load into Input Tensor
    for (int i = 0; i < 832; i++) {
        float incoming_sample_normalized = (inputBuff_pointer[i] - mean) / (std_ + 1e-8);
        input->data.int8[i] = (int8_t)(incoming_sample_normalized / input->params.scale + input->params.zero_point);
    }

    // 2. Run Inference
    interpreter->Invoke();

    // 3. Find ArgMax
    int pred_class = 0;
    float max_prob = -1.0;
    for (int i = 0; i < 3; i++) {
        float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        if (prob > max_prob) {
            max_prob = prob;
            pred_class = i;
        }
    }

    // 4. Send Result back to Python
    Serial.printf("RESULT:%d,%.4f\n", pred_class, max_prob);

    return pred_class;
}