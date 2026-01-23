#pragma once

#include <Arduino.h>

// CONSTANTES
#define microphonePin 34
#define SAMPLE_RATE 12000
#define RECORD_TIME 2 // segundos
const unsigned int TOTAL_SAMPLES = (SAMPLE_RATE * RECORD_TIME);
// Reservamos el buffer en la memoria del ESP32
extern float audioBuffer[TOTAL_SAMPLES];
extern bool isRecording;


// DECLARACIÓN DE FUNCIONES
bool setup_AudioRecording();
bool recordAudio(float* audioBuffer);
void playBackSerial(float* audioBuffer, int len);