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
    // 1. Quantize and Load the real data into the input tensor
    // The test_sample_data
    for (int i = 0; i < 832; i++) {
        float raw_val = sample_input[i];
        
        // Formula: quantized_val = (float_val / scale) + zero_point
        int8_t quantized_val = (int8_t)(raw_val / input->params.scale + input->params.zero_point);
        
        input->data.int8[i] = quantized_val;
    }

    // 2. Run Inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

    // 1. Check if the input is actually changing
    Serial.printf("Input[0] raw: %f, quantized: %d\n", sample_input[0], input->data.int8[0]);

    // 2. Check the raw output bytes before the math
    Serial.printf("Raw Output Bytes: %d, %d, %d\n", 
                output->data.int8[0], 
                output->data.int8[1], 
                output->data.int8[2]);

    // 3. Print the scales (if these are 0, your math will always be 0)
    // Check if params are actually zero and try to recover them
    float output_scale = output->params.scale;
    int output_zp = output->params.zero_point;

    // If the library is failing to report the scale in the struct, 
    Serial.printf("Output Scale: %f, ZeroPoint: %d\n", output->params.scale, output->params.zero_point);
    
    // check the quantization object directly:
    if (output_scale == 0.0f) {
        auto* quant = reinterpret_cast<TfLiteAffineQuantization*>(output->quantization.params);
        if (quant && quant->scale) {
            output_scale = quant->scale->data[0];
            output_zp = quant->zero_point->data[0];
        }
        Serial.printf("Output Scale: %f, ZeroPoint: %d\n", output_scale, output_zp);
    }


    // 3. Print Results
    Serial.print("Model Output: ");
    for (int i = 0; i < 3; i++) {
        float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        Serial.print(prob, 4);
        if (i < 2) Serial.print(", ");
    }
    Serial.println("\n");

    delay(5000); // Wait 5 seconds between tests
}