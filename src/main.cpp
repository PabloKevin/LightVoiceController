#include <Arduino.h>

#include "model.h"
#include "audio_recording.h"
#include "audio_preProcessing.h"

// --- Configuration ---
#define potentionmeterPin 35
//pinMode(potentionmeterPin, INPUT);
AudioProcessor processor;

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(2000);

    if (!setup_AudioRecording()){
        Serial.println("Error during AudioRecording setup");
    }
    if (!setup_MLmodel()){
        Serial.println("Error during ML model setup");
    }
    
}

void loop() {
    if (Serial.available() > 0) {
        char c = Serial.read();
        if (c == 'g') {
            digitalWrite(LED_BUILTIN, HIGH);
            recordAudio();

            processor.process_complete_pipeline(audioBuffer, TOTAL_SAMPLES);
            digitalWrite(LED_BUILTIN, LOW);
            playBackSerial();
        }
    }
}




