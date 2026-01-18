#include <Arduino.h>
#include "model_data.h" // Ensure this defines g_model_data (or rename below)

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// 1. Setup Resolver: Increase the number (currently <10>) if your model uses 
// more types of layers (Conv2D, Pooling, etc.). 
using MyOpResolver = tflite::MicroMutableOpResolver<10>;

// Global variables for TFLM
const int kTensorArenaSize = 80 * 1024; // Increased to be safe
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    tflite::InitializeTarget();

    // 2. Load Model
    model = tflite::GetModel(voiceModel_3classes_tflite); 
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model version mismatch!");
        return;
    }

    // 3. Register Operations
    // Note: If you get "Did you use the wrong OpResolver?" errors, 
    // add the missing ops here (e.g., op_resolver.AddConv2D()).
    static tflite::AllOpsResolver op_resolver;
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddReshape(); // Common for (13, 14, 1) inputs

    // 4. Instantiate Interpreter
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // 5. Allocate Tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return;
    }

    // 6. Assign input/output pointers
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Model Initialized Successfully");
}

void loop() {
    // Input is (13, 14, 1) = 182 bytes
    // For demonstration, we fill it with zeros or dummy data
    for (int i = 0; i < 182; i++) {
        // Applying quantization formula if you have float data:
        // float val = 0.5f;
        // input->data.int8[i] = (int8_t)(val / input->params.scale + input->params.zero_point);
        input->data.int8[i] = 0; 
    }

    // 7. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference Invoke failed!");
        return;
    }

    // 8. Print 3 Probability Outputs
    Serial.print("Inference Result: ");
    for (int i = 0; i < 3; i++) {
        // De-quantize the int8 output to a 0.0 - 1.0 float
        float probability = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        Serial.print(probability, 4);
        if (i < 2) Serial.print(", ");
    }
    Serial.println();

    delay(2000);
}