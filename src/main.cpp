#include <Arduino.h>
#include "audio_provider.h"
#include "mfcc.h"
#include "feature_provider.h"
#include "tflite_inference.h"
#include "wifi_control.h"
#include "audio_config.h"

void setup() {
    Serial.begin(115200);

    audio_init();
    mfcc_init();
    wifi_init();

    if (tflite_init() != 0) {
        Serial.println("TFLite init failed");
        while (1);
    }

    Serial.println("Voice controller ready");
}

void loop() {
    static int16_t audio_frame[FRAME_SAMPLES];
    static float mfcc_frame[MFCC_COEFFS];
    static float features[MFCC_FRAMES * MFCC_COEFFS];
    static float output[10];

    audio_read_frame(audio_frame);
    mfcc_compute(audio_frame, mfcc_frame);

    if (feature_add_frame(mfcc_frame)) {
        feature_get(features);

        if (tflite_run(features, output) == 0) {
            int cmd = 0;
            float best = 0;
            for (int i = 0; i < 10; i++) {
                if (output[i] > best) {
                    best = output[i];
                    cmd = i;
                }
            }

            if (cmd == 1) light_on();
            if (cmd == 2) light_off();
        }

        feature_reset();
    }
}
