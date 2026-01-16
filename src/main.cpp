#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"
#include "audio_processing.h"
#include "mfcc.h"
#include "tflite_inference.h"
#include "model_data.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// --- Configuración ---
#define MICROPHONE_PIN 34
#define POTENTIOMETER_PIN 35

// --- Objetos y Variables Globales ---
float* audioBuffer = nullptr; 
float mfcc_normalized[N_MFCC * MAX_TIME_STEPS];
bool isModelInitialized = false;

AudioProcessor audioProcessor;
MFCCExtractor mfccExtractor;
VoiceModelInference voiceModel;
WiFiUDP udp;

QueueHandle_t audioReadyQueue;
SemaphoreHandle_t modelReadySemaphore; // Declaración global
char bedroomLightIP[16] = "";

// --- PROTOTIPOS DE FUNCIONES (Para que el compilador las conozca de antemano) ---
void wifiAndAudioTask(void* pvParameters);
void mlInferenceTask(void* pvParameters);
void handlePrediction(int classId, float confidence);
void recordAudio();
void turnOn();
void turnOff();
void sendWizCommand(const char* jsonCommand);

// ============================================================================
// Tarea de Inferencia (Corregida)
// ============================================================================
void mlInferenceTask(void* pvParameters) {
    float mfcc_temp[N_MFCC * MAX_TIME_STEPS];

    while (1) {
        if (xQueueReceive(audioReadyQueue, nullptr, portMAX_DELAY)) {
            Serial.println("\n=== [ML] Procesando ===");
            
            audioProcessor.process_complete_pipeline(audioBuffer, TOTAL_SAMPLES);
            int num_frames = mfccExtractor.extract_mfcc(audioBuffer, TOTAL_SAMPLES, mfcc_temp, MAX_TIME_STEPS);
            mfccExtractor.pad_to_fixed_length(mfcc_temp, num_frames, mfcc_normalized, MAX_TIME_STEPS);
            
            if (audioBuffer != nullptr) {
                free(audioBuffer);
                audioBuffer = nullptr; 
                Serial.println("♻️ Audio liberado");
            }

            vTaskDelay(50 / portTICK_PERIOD_MS);

            if (!isModelInitialized) {
                isModelInitialized = voiceModel.initialize(voiceModel_3classes_tflite);
            }

            if (isModelInitialized) {
                int classId;
                float confidence;
                voiceModel.predict_with_confidence(classId, confidence, mfcc_normalized);
                
                if (xSemaphoreTake(modelReadySemaphore, portMAX_DELAY)) {
                    handlePrediction(classId, confidence);
                    xSemaphoreGive(modelReadySemaphore);
                }
            }
        }
    }
}

// ============================================================================
// Lógica de Predicción y Control
// ============================================================================
void handlePrediction(int classId, float confidence) {
    const char* name = voiceModel.get_class_name(classId);
    Serial.printf("🔍 Predicción: %s (%.2f%%)\n", name, confidence * 100);
    
    if (confidence > 0.75f) {
        if (classId == CLASS_APAGAR_LUZ) turnOff();
        else if (classId == CLASS_PRENDER_LUZ) turnOn();
    }
}

void sendWizCommand(const char* jsonCommand) {
    if (bedroomLightIP[0] == '\0') return;
    udp.beginPacket(bedroomLightIP, 38899);
    udp.print(jsonCommand);
    udp.endPacket();
}

void turnOn() { sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":true}}"); }
void turnOff() { sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":false}}"); }

// ============================================================================
// WiFi y Grabación
// ============================================================================
void recordAudio() {
    Serial.println("🎙️ Grabando...");
    digitalWrite(LED_BUILTIN, HIGH);
    unsigned int sampleDelay = 1000000 / SAMPLE_RATE;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long next = micros() + sampleDelay;
        audioBuffer[i] = (analogRead(MICROPHONE_PIN) / 2048.0f) - 1.0f;
        while (micros() < next);
    }
    digitalWrite(LED_BUILTIN, LOW);
}

void wifiAndAudioTask(void* pvParameters) {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) { vTaskDelay(500 / portTICK_PERIOD_MS); }
    Serial.println("✅ WiFi Conectado");
    udp.begin(38899);

    while (1) {
        if (Serial.available() > 0 && Serial.read() == 'g') {
            audioBuffer = (float*)heap_caps_malloc(TOTAL_SAMPLES * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (audioBuffer != nullptr) {
                recordAudio();
                xQueueSend(audioReadyQueue, nullptr, portMAX_DELAY);
            }
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

// ============================================================================
// Setup y Loop
// ============================================================================
void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    analogReadResolution(12);
    
    audioReadyQueue = xQueueCreate(1, 0);
    modelReadySemaphore = xSemaphoreCreateMutex(); // Inicialización del semáforo

    xTaskCreatePinnedToCore(wifiAndAudioTask, "WiFiTask", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(mlInferenceTask, "MLTask", 8192, NULL, 1, NULL, 1);

    Serial.println("=== Sistema Listo ===");
}

void loop() { vTaskDelay(portMAX_DELAY); }