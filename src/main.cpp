#include <Arduino.h>
#include "model_data.h" // Ensure this defines g_model_data (or rename below)
#include "sample_data.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
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
    Serial.setTimeout(2000);
    
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
        return;
    }

    // 6. Assign input/output pointers
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Model Initialized Successfully");
}

void loop() {
    if (Serial.available() > 0) {
        char start_cmd = Serial.read();
        
        if (start_cmd == 'S') {
            float incoming_sample[832];
            
            // Read 832 floats (3328 bytes)
            // Note: readBytes is blocking, ensure timeout is sufficient
            uint8_t* buffer = (uint8_t*)incoming_sample;
            size_t bytes_to_read = 832 * sizeof(float);
            size_t bytes_read = Serial.readBytes(buffer, bytes_to_read);

            if (bytes_read == bytes_to_read) {
                // 1. Quantize and Load into Input Tensor
                for (int i = 0; i < 832; i++) {
                    input->data.int8[i] = (int8_t)(incoming_sample[i] / input->params.scale + input->params.zero_point);
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
            }
        }
    }
}