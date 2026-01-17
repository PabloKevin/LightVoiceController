#pragma once
#include <stdbool.h>

bool feature_add_frame(const float* mfcc_frame);
void feature_get(float* out);
void feature_reset();
