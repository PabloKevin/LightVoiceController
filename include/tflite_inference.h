#pragma once
#include "tensorflow/lite/c/common.h"

bool tflite_init();
bool tflite_invoke();

TfLiteTensor* tflite_input();
TfLiteTensor* tflite_output();
int tflite_run(const float* input, float* output);
