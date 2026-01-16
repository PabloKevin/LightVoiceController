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

// --- Buffers Dinámicos y Globales ---
float* audioBuffer = nullptr; // Se asignará dinámicamente
float mfcc_normalized[N_MFCC * MAX_TIME_STEPS];

AudioProcessor audioProcessor;
MFCCExtractor mfccExtractor;
VoiceModelInference voiceModel;
WiFiUDP udp;

QueueHandle_t audioReadyQueue;
char bedroomLightIP[16] = "";
bool isModelInitialized = false;

// --- Prototipos ---
void recordAudio();
void turnOn();
void turnOff();

// ============================================================================
// Control WiZ
// ============================================================================

void sendWizCommand(const char* jsonCommand) {
    if (bedroomLightIP[0] == '\0') return;
    udp.beginPacket(bedroomLightIP, 38899);
    udp.print(jsonCommand);
    udp.endPacket();
}

void turnOn() { sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":true}}"); }
void turnOff() { sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":false}}"); }

// ============================================================================
// Tareas FreeRTOS
// ============================================================================

void wifiAndAudioTask(void* pvParameters) {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) { vTaskDelay(500 / portTICK_PERIOD_MS); }
    Serial.println("✅ WiFi Conectado");
    udp.begin(38899);

    while (1) {
        if (Serial.available() > 0 && Serial.read() == 'g') {
            // 1. Reservar memoria para el audio (96KB)
            audioBuffer = (float*)heap_caps_malloc(TOTAL_SAMPLES * sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            
            if (audioBuffer != nullptr) {
                recordAudio();
                // 2. Avisar a la tarea de ML
                xQueueSend(audioReadyQueue, nullptr, portMAX_DELAY);
            } else {
                Serial.println("❌ RAM insuficiente para grabar");
            }
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

void mlInferenceTask(void* pvParameters) {
    float mfcc_temp[N_MFCC * MAX_TIME_STEPS];

    while (1) {
        if (xQueueReceive(audioReadyQueue, nullptr, portMAX_DELAY)) {
            Serial.println("⚙️ Procesando Audio...");
            
            // 1. Pre-procesamiento
            audioProcessor.process_complete_pipeline(audioBuffer, TOTAL_SAMPLES);
            
            // 2. Extraer MFCC
            int num_frames = mfccExtractor.extract_mfcc(audioBuffer, TOTAL_SAMPLES, mfcc_temp, MAX_TIME_STEPS);
            mfccExtractor.pad_to_fixed_length(mfcc_temp, num_frames, mfcc_normalized, MAX_TIME_STEPS);
            
            // 3. ¡LIBERAR MEMORIA! (Crucial)
            // Borramos el buffer de audio antes de llamar a TensorFlow
            free(audioBuffer);
            audioBuffer = nullptr; 
            Serial.println("♻️ Memoria de audio liberada. Iniciando TFLite...");

            // 4. Inferencia (Inicializar si es la primera vez o cada vez para ahorrar)
            // Si el error persiste, mueve el voiceModel.initialize aquí adentro
            if (!isModelInitialized) {
                isModelInitialized = voiceModel.initialize(voiceModel_3classes_tflite);
            }

            if (isModelInitialized) {
                int classId;
                float confidence;
                voiceModel.predict_with_confidence(classId, confidence, mfcc_normalized);
                
                Serial.printf("Predicción: %s (%.2f%%)\n", voiceModel.get_class_name(classId), confidence * 100);
                
                if (confidence > 0.70f) {
                    if (classId == CLASS_APAGAR_LUZ) turnOff();
                    if (classId == CLASS_PRENDER_LUZ) turnOn();
                }
            }
            Serial.println(" esperando comando 'g'...");
        }
    }
}

// ============================================================================
// Lógica de Grabación
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

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    analogReadResolution(12);
    
    audioReadyQueue = xQueueCreate(1, 0);

    // No inicializamos el modelo aquí para no saturar la RAM desde el arranque
    Serial.println("=== Sistema Listo. Presiona 'g' para comando de voz ===");

    xTaskCreatePinnedToCore(wifiAndAudioTask, "WiFiTask", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(mlInferenceTask, "MLTask", 8192, NULL, 1, NULL, 1);
}

void loop() { vTaskDelay(portMAX_DELAY); }