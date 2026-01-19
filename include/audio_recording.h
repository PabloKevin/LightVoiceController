#pragma once

#include <Arduino.h>

// CONSTANTES
#define microphonePin 34
#define SAMPLE_RATE 12000
#define RECORD_TIME 0.5 // segundos
const unsigned int TOTAL_SAMPLES = (SAMPLE_RATE * RECORD_TIME);
// Reservamos el buffer en la memoria del ESP32
union audio {
    unsigned int raw[TOTAL_SAMPLES];      // Para la grabación (24,000 bytes)
    float processed[TOTAL_SAMPLES]; // Para el procesamiento (24,000 bytes)
};
extern audio audioBuffer[TOTAL_SAMPLES];
extern bool isRecording;


// DECLARACIÓN DE FUNCIONES
bool setup_AudioRecording();
bool recordAudio();
void playBackSerial();