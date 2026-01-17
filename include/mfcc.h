#pragma once
#include <stdint.h>

#define MFCC_COEFFS 13

void mfcc_init();
void mfcc_compute(const int16_t* audio, float* mfcc_out);
