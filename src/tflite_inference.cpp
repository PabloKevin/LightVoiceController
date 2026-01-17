#include "tflite_inference.h"

#include "model_data.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include <Arduino.h>

// =====================
// TFLite globals
// =====================
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor = nullptr;

  // ⚠️ Tune this if needed (80 KB is safe for small CNNs)
  constexpr int kTensorArenaSize = 80 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// =====================
// Initialization
// =====================
bool tflite_init() {
  // Load model
  model = tflite::GetModel(voiceModel_3classes_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model schema %d not equal to supported %d",
      model->version(), TFLITE_SCHEMA_VERSION
    );
    return false;
  }

  // Resolver (use AllOpsResolver for safety)
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return false;
  }

  // Get input & output tensors
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Debug info
  Serial.println("🧠 TFLite initialized");
  Serial.print("Input type: ");
  Serial.println(input_tensor->type);
  Serial.print("Input bytes: ");
  Serial.println(input_tensor->bytes);
  Serial.print("Output bytes: ");
  Serial.println(output_tensor->bytes);

  return true;
}

// =====================
// Run inference
// =====================
bool tflite_invoke() {
  if (!interpreter) {
    return false;
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return false;
  }

  return true;
}

// =====================
// Access tensors
// =====================
TfLiteTensor* tflite_input() {
  return input_tensor;
}

TfLiteTensor* tflite_output() {
  return output_tensor;
}

int tflite_run(const float* input, float* output) {
  TfLiteTensor* in = tflite_input();
  TfLiteTensor* out = tflite_output();

  memcpy(in->data.f, input, in->bytes);

  if (!tflite_invoke()) {
    return -1;
  }

  memcpy(output, out->data.f, out->bytes);
  return 0;
}
