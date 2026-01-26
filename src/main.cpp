#include <Arduino.h>

#include "model.h"
#include "audio_recording.h"
#include "audio_preProcessing.h"
#include "mfcc.h"

// --- Configuration ---
#define potentionmeterPin 35
//pinMode(potentionmeterPin, INPUT);
AudioProcessor processor;

// Handlers para las tareas (opcional, sirve para controlarlas luego)
TaskHandle_t AudioRecording_task;
TaskHandle_t AudioProccessing_task;
TaskHandle_t Prediction_task;

bool start_recording = false;
bool copying = false;
bool isRecording = false;

void code_AudioRecording(void * parameter);
void code_AudioProccessing(void * parameter);
void code_Prediction(void * parameter);
void printMemoryStats();

QueueHandle_t audioQueue = NULL;
QueueHandle_t inputQueue = NULL;

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(2000);

    audioQueue = xQueueCreate(1, sizeof(float*));
    inputQueue = xQueueCreate(1, sizeof(float*));

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
        12000,
        NULL,
        1,
        &AudioProccessing_task,
        0);                 /* <--- NÚCLEO 1 */
    // Crear Tarea 2: Procesamiento (En el Núcleo 1)
    xTaskCreatePinnedToCore(
        code_Prediction,
        "TareaPrediccion",
        5000,
        NULL,
        1,
        &Prediction_task,
        1);                 /* <--- NÚCLEO 1 */
    
    Serial.println("\nEND SETUP");
    printMemoryStats();
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

                //Serial.println("\nDespués de grabar, previo a enviar audioBuffer por la Queue");
                //printMemoryStats();
                playBackSerial(audioBuffer, TOTAL_SAMPLES);

                xQueueSend(audioQueue, &audioBuffer, portMAX_DELAY); // Enviar dirección
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void code_AudioProccessing(void * parameter){
    for(;;) {
        float* audioBuffer_ptr;
        bool flagPredict = true;
        if (audioQueue != NULL) {
            if (xQueueReceive(audioQueue, &audioBuffer_ptr, portMAX_DELAY)) {
                if (copying && !isRecording){
                    copying = false;
                    processor.process_complete_pipeline(audioBuffer_ptr, TOTAL_SAMPLES);

                    Serial.println("processedAudio");
                    playBackSerial(audioBuffer_ptr, working_len);

                    //Serial.println("\nPost Procesamiento de audio");
                    //printMemoryStats();

                    float* input_MFCC = new float[13*64];
                    MFCCExtractor mfcc;
                    mfcc.extract_mfcc(audioBuffer_ptr, working_len, input_MFCC);
                    Serial.println("MFCC extaridos correctamente");
                    Serial.println("MFCC");
                    for (int i = 0; i < 13 * 64; i++) {
                        Serial.println(input_MFCC[i], 6); 
                    }
                    Serial.println("Finish");
                    //Serial.println("\nPost extracción de MFCCs");
                    //printMemoryStats();
                    delete[] audioBuffer_ptr; // LIBERAR RAM AQUÍ
                    //Serial.println("\nPost eliminación de audioBuffer");
                    //printMemoryStats();
                    xQueueSend(inputQueue, &input_MFCC, portMAX_DELAY);
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void code_Prediction(void * parameter){
    for(;;) {
        float* input_MFCC_ptr;
        if (inputQueue != NULL) {
            if (xQueueReceive(inputQueue, &input_MFCC_ptr, portMAX_DELAY)) {
                unsigned long startTime = millis();
                if (!setup_MLmodel()){
                    Serial.println("Error during ML model setup");
                }
                int action = predict(input_MFCC_ptr);
                //Serial.println("\nPost predicción");
                //printMemoryStats();

                free_MLmodel();
                delete[] input_MFCC_ptr;

                //Serial.println("\nPost eliminación de ML model e input_MFCC");
                //printMemoryStats();
                Serial.printf(">>> Predicción finalizada. Tiempo total: %lu ms\n", millis() - startTime);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}


void printMemoryStats() {
    // 1. Memoria total libre actualmente
    size_t freeHeap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    
    // 2. El bloque más grande que puedes pedir con un solo 'new' o 'malloc'
    // Si esto es menor que tus 72KB del modelo, el código fallará.
    size_t largestBlock = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    
    // 3. El punto más bajo de RAM desde que se encendió (High Water Mark)
    // Útil para saber qué tan cerca estuviste del colapso en el momento de más estrés.
    size_t minFree = esp_get_minimum_free_heap_size();

    Serial.println("--- ESTADO DE LA MEMORIA RAM ---");
    Serial.printf("Total libre:      %d bytes (%.2f KB)\n", freeHeap, freeHeap / 1024.0);
    Serial.printf("Bloque + grande:  %d bytes (%.2f KB)\n", largestBlock, largestBlock / 1024.0);
    Serial.printf("Mínima histórica: %d bytes (%.2f KB)\n", minFree, minFree / 1024.0);
    Serial.println("--------------------------------");
}