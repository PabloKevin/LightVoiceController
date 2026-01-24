#pragma once

#include <Arduino.h>
#include "model_data.h" // Model weights

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Constantes: 
// 1. Setup Resolver: Increase the number (currently <10>) if your model uses 
// more types of layers (Conv2D, Pooling, etc.). 

// Configuración
extern const int kTensorArenaSize;
extern const float mean;
extern const float std_;

// Punteros a los objetos de TensorFlow (Declaración externa)
extern const tflite::Model* model;
extern tflite::MicroInterpreter* interpreter;
extern TfLiteTensor* input;
extern TfLiteTensor* output;


// Declaración de funciones (solo la firma)
bool setup_MLmodel();
int predict(float* inputBuff_pointer);
void free_MLmodel();