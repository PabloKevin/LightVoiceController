#include "audio_input.h"
#include <Arduino.h>

static int g_adc_pin;

void audio_init(int adc_pin) {
    g_adc_pin = adc_pin;
    analogReadResolution(12);
}

void audio_read_frame(int16_t* buffer) {
    for (int i = 0; i < FRAME_SAMPLES; i++) {
        int raw = analogRead(g_adc_pin);
        buffer[i] = (int16_t)(raw - 2048);  // center
        delayMicroseconds(1000000 / SAMPLE_RATE);
    }
}
