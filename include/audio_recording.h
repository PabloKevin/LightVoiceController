#pragma once

#include <Arduino.h>

// CONSTANTES
#define microphonePin 34
#define SAMPLE_RATE 12000
#define RECORD_TIME 1 // segundos
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_TIME)
// Reservamos el buffer en la memoria del ESP32
extern uint16_t audioBuffer[TOTAL_SAMPLES]; 
extern bool isRecording;


// DECLARACIÓN DE FUNCIONES
bool setup_AudioRecording();
bool recordAudio();
void playBackSerial();

