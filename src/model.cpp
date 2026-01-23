#include "model.h"

// Definición de constantes
const int kTensorArenaSize = 70 * 1024;
const float mean = -50.281225f;
const float std_ = 159.534246f;

bool setup_MLmodel(){
    // Definición de variables globales (Aquí se les asigna memoria)
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    tflite::InitializeTarget();

    // 2. Load Model
    model = tflite::GetModel(voiceModel_3classes_tflite); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model version mismatch!");
        return 0;
    }

    // 3. Register Operations
    // Note: If you get "Did you use the wrong OpResolver?" errors, 
    // add the missing ops here (e.g., op_resolver.AddConv2D()).
    static tflite::AllOpsResolver op_resolver;
    //op_resolver.AddFullyConnected();
    //op_resolver.AddSoftmax();
    //op_resolver.AddReshape(); // Common for (13, 14, 1) inputs

    // 4. Instantiate Interpreter
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // 5. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return 0;
    }

    // 6. Assign input/output pointers
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Model Initialized Successfully");
    return 1;
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