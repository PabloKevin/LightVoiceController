#pragma once
#include <stdint.h>

#define SAMPLE_RATE     12000
#define FRAME_MS        20
#define FRAME_SAMPLES   (SAMPLE_RATE * FRAME_MS / 1000)

void audio_init(int adc_pin);
void audio_read_frame(int16_t* buffer);
