#include "audio_recording.h"

// Reservamos el buffer en la memoria del ESP32
float audioBuffer[TOTAL_SAMPLES]; 
bool isRecording = false;


bool setup_AudioRecording(){
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(microphonePin, INPUT);
    analogReadResolution(12); // Máxima resolución del ADC
    return 1;
}


bool recordAudio() {
    Serial.println(">>> Iniciando grabación (2 seg)...");
    
    // Calculamos el tiempo entre muestras en microsegundos
    unsigned int sampleDelay = 1000000 / SAMPLE_RATE;
    unsigned long startTime = millis();

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long nextSampleTime = micros() + sampleDelay;
        
        audioBuffer[i] = (-(float)(microphonePin) + 2048) * 16;
        
        // Esperamos con precisión de microsegundos para mantener la frecuencia
        while (micros() < nextSampleTime) {
            // Espera activa para precisión
        }
    }

    Serial.printf(">>> Grabación finalizada. Tiempo total: %lu ms\n", millis() - startTime);

    return 1;
}

// Función para enviar los datos al Serial y graficarlos (Serial Plotter)
void playBackSerial() {
    Serial.println("Enviando datos de audio al monitor...");
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        Serial.println((int)(audioBuffer[i]*32767));
        // delayMicroseconds(100); // Opcional, para no saturar el buffer del PC
    }
}
