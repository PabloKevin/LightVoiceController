#include <Arduino.h>

#include "model.h"
#include "audio_recording.h"
#include "audio_preProcessing.h"

// --- Configuration ---
#define potentionmeterPin 35
//pinMode(potentionmeterPin, INPUT);
AudioProcessor processor;

// Handlers para las tareas (opcional, sirve para controlarlas luego)
TaskHandle_t AudioRecording_task;
TaskHandle_t AudioProccessing_task;

bool start_recording = false;
bool copying = false;
bool isRecording = false;

void code_AudioRecording(void * parameter);
void code_AudioProccessing(void * parameter);

QueueHandle_t audioQueue = xQueueCreate(1, sizeof(float*));


void setup() {
    Serial.begin(115200);
    Serial.setTimeout(2000);

    if (!setup_AudioRecording()){
        Serial.println("Error during AudioRecording setup");
    }
    
    //if (!setup_MLmodel()){
    //    Serial.println("Error during ML model setup");
    //}

    // Crear Tarea 1: Grabación (En el Núcleo 0)
    xTaskCreatePinnedToCore(
        code_AudioRecording,   /* Función que contiene el código */
        "TareaAudio",       /* Nombre de la tarea */
        10000,              /* Tamaño de la pila (Stack size) en bytes */
        NULL,               /* Parámetros de entrada */
        1,                  /* Prioridad de la tarea (0 es la más baja) */
        &AudioRecording_task,        /* Handler */
        0);                 /* <--- NÚCLEO 0 */

    // Crear Tarea 2: Procesamiento (En el Núcleo 1)
    xTaskCreatePinnedToCore(
        code_AudioProccessing,
        "TareaProcesamiento",
        10000,
        NULL,
        1,
        &AudioProccessing_task,
        1);                 /* <--- NÚCLEO 1 */
    
}

void loop() {

}


void code_AudioRecording(void * parameter){
    for(;;) {
        if (Serial.available() > 0) {
            char c = Serial.read();
            if (c == 'g') {
                float* audioBuffer = new float[TOTAL_SAMPLES];
                isRecording = true;
                digitalWrite(LED_BUILTIN, HIGH);
                recordAudio(audioBuffer);
                digitalWrite(LED_BUILTIN, LOW);
                isRecording = false;
                copying=true;
                xQueueSend(audioQueue, &audioBuffer, portMAX_DELAY); // Enviar dirección
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void code_AudioProccessing(void * parameter){
    for(;;) {
        float* audioBuffer;
        if (xQueueReceive(audioQueue, &audioBuffer, portMAX_DELAY)) {
            if (copying && !isRecording){
                copying = false;
                processor.process_complete_pipeline(audioBuffer, TOTAL_SAMPLES);
                playBackSerial(audioBuffer, working_len);
            }
            delete[] audioBuffer; // LIBERAR RAM AQUÍ
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    float* frameRecibido;
    
}