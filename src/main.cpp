#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"
#include "audio_processing.h"
#include "mfcc.h"
#include "tflite_inference.h"
#include "model_data.h"  // Modelo TFLite embebido
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"

// --- Configuración de Pines ---
#define MICROPHONE_PIN 34
#define POTENTIOMETER_PIN 35

// --- Buffers de Audio y Procesamiento ---
// TOTAL_SAMPLES, N_MFCC y MAX_TIME_STEPS se definen en audio_processing.h
float audioBuffer[TOTAL_SAMPLES]; // Un solo buffer para ahorrar RAM
float mfcc_normalized[N_MFCC * MAX_TIME_STEPS];

// --- Objetos Globales ---
AudioProcessor audioProcessor;
MFCCExtractor mfccExtractor;
VoiceModelInference voiceModel;

// --- WiFi y Control de Luces WiZ ---
WiFiUDP udp;
char bedroomLightIP[16] = "";
const int wizPort = 38899;
int currentBrightness = 50;
int currentTemp = 4100;

// --- Colas y Semáforos FreeRTOS ---
QueueHandle_t audioReadyQueue;  
SemaphoreHandle_t modelReadySemaphore;  

// --- Handles de Tareas ---
TaskHandle_t wifiTaskHandle = nullptr;
TaskHandle_t mlTaskHandle = nullptr;

bool modelReady = false;

// --- Prototipos de Funciones ---
void wifiAndAudioTask(void* pvParameters);
void mlInferenceTask(void* pvParameters);
bool findLightIP(const char* targetMAC, char* targetIP);
void sendWizCommand(const char* jsonCommand, char* targetIP);
void turnOn();
void turnOff();
void setWizLight(int brightness, int kelvin, int sceneID, char* targetIP);
void Pot2Light();
void recordAudio();
void handlePrediction(int classId, float confidence);

// ============================================================================
// Funciones de Control WiZ
// ============================================================================

bool findLightIP(const char* targetMAC, char* targetIP) {
    Serial.println("Buscando lámpara WiZ...");
    IPAddress broadcastIP(255, 255, 255, 255);
    const char* discoverCmd = "{\"method\":\"getSystemConfig\",\"params\":{}}";
    
    udp.beginPacket(broadcastIP, wizPort);
    udp.print(discoverCmd);
    udp.endPacket();

    unsigned long startMillis = millis();
    while (millis() - startMillis < 2000) {
        int packetSize = udp.parsePacket();
        if (packetSize) {
            char incomingPacket[512];
            int len = udp.read(incomingPacket, 511);
            incomingPacket[len] = '\0';
            JsonDocument doc;
            DeserializationError error = deserializeJson(doc, incomingPacket);
            if (!error) {
                const char* mac = doc["result"]["mac"];
                if (mac && strcasecmp(mac, targetMAC) == 0) {
                    strncpy(targetIP, udp.remoteIP().toString().c_str(), 15);
                    targetIP[15] = '\0';
                    Serial.printf("¡Lámpara encontrada! IP: %s\n", targetIP);
                    return true;
                }
            }
        }
    }
    return false;
}

void sendWizCommand(const char* jsonCommand, char* targetIP) {
    if (targetIP[0] == '\0') return;
    udp.beginPacket(targetIP, wizPort);
    udp.print(jsonCommand);
    udp.endPacket();
}

void turnOn() {
    sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":true}}", bedroomLightIP);
}

void turnOff() {
    sendWizCommand("{\"method\":\"setPilot\",\"params\":{\"state\":false}}", bedroomLightIP);
}

void setWizLight(int brightness, int kelvin, int sceneID, char* targetIP) {
    if (targetIP[0] == '\0') return;
    brightness = constrain(brightness, 10, 100);
    kelvin = constrain(kelvin, 2700, 6500);
    JsonDocument doc;
    doc["method"] = "setPilot";
    JsonObject params = doc["params"].to<JsonObject>();
    params["state"] = true;
    params["dimming"] = brightness;
    params["temp"] = kelvin;
    params["sceneID"] = sceneID;
    char buffer[128];
    serializeJson(doc, buffer);
    sendWizCommand(buffer, targetIP);
}

void Pot2Light() {
    static int lastPotValue = 0;
    int newPotValue = analogRead(POTENTIOMETER_PIN);
    if (abs(newPotValue - lastPotValue) < 100) return;
    lastPotValue = newPotValue;

    if (newPotValue < 150) {
        turnOff();
    } else {
        currentBrightness = map(newPotValue, 150, 4095, 10, 100);
        currentTemp = map(newPotValue, 150, 4095, 2700, 6500);
        setWizLight(currentBrightness, currentTemp, 6, bedroomLightIP);
    }
}

// ============================================================================
// Grabación de Audio
// ============================================================================

void recordAudio() {
    Serial.println(">>> Grabando (2 seg)...");
    digitalWrite(LED_BUILTIN, HIGH);
    unsigned int sampleDelay = 1000000 / SAMPLE_RATE;
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long nextSampleTime = micros() + sampleDelay;
        uint16_t adcValue = analogRead(MICROPHONE_PIN);
        audioBuffer[i] = (adcValue / 2048.0f) - 1.0f;
        while (micros() < nextSampleTime);
    }
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println(">>> Grabación finalizada.");
}

void handlePrediction(int classId, float confidence) {
    Serial.printf("Predicción: %s (Confianza: %.2f%%)\n", voiceModel.get_class_name(classId), confidence * 100);
    
    if (confidence < 0.75f) {
        Serial.println("Confianza baja, ignorando...");
        return;
    }

    if (classId == CLASS_APAGAR_LUZ) {
        Serial.println("Acción: Apagar Luz");
        turnOff();
    } else if (classId == CLASS_PRENDER_LUZ) {
        Serial.println("Acción: Prender Luz");
        turnOn();
    } else if (classId == CLASS_AMBIENTE) {
        Serial.println("Acción: Modo Ambiente");
        setWizLight(30, 3000, 6, bedroomLightIP);
    }
}

// ============================================================================
// Tareas FreeRTOS
// ============================================================================

void wifiAndAudioTask(void* pvParameters) {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        vTaskDelay(500 / portTICK_PERIOD_MS);
        Serial.print(".");
    }
    Serial.println("\nWiFi Conectado");
    udp.begin(wizPort);
    
    while (!findLightIP(bedroom_light_mac, bedroomLightIP)) {
        vTaskDelay(3000 / portTICK_PERIOD_MS);
    }

    while (1) {
        if (Serial.available() > 0 && Serial.read() == 'g') {
            recordAudio();
            xQueueSend(audioReadyQueue, nullptr, portMAX_DELAY);
        }
        Pot2Light();
        vTaskDelay(150 / portTICK_PERIOD_MS);
    }
}

void mlInferenceTask(void* pvParameters) {
    while (!modelReady) vTaskDelay(100 / portTICK_PERIOD_MS);
    float mfcc_temp[N_MFCC * MAX_TIME_STEPS]; // Buffer temporal para extracción

    while (1) {
        if (xQueueReceive(audioReadyQueue, nullptr, portMAX_DELAY)) {
            unsigned long start = millis();
            
            // 1. Pre-procesamiento de audio (Normalización, Filtros, Kill Peaks)
            audioProcessor.process_complete_pipeline(audioBuffer, TOTAL_SAMPLES);
            
            // 2. Extracción de MFCC
            int num_frames = mfccExtractor.extract_mfcc(audioBuffer, TOTAL_SAMPLES, mfcc_temp, MAX_TIME_STEPS);
            
            // 3. Ajuste a longitud fija y normalización MFCC
            mfccExtractor.pad_to_fixed_length(mfcc_temp, num_frames, mfcc_normalized, MAX_TIME_STEPS);
            
            // 4. Inferencia TFLite
            int classId;
            float confidence;
            voiceModel.predict_with_confidence(classId, confidence, mfcc_normalized);
            
            Serial.printf("Tiempo de proceso: %lu ms\n", millis() - start);
            
            xSemaphoreTake(modelReadySemaphore, portMAX_DELAY);
            handlePrediction(classId, confidence);
            xSemaphoreGive(modelReadySemaphore);
        }
    }
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    analogReadResolution(12);
    
    audioReadyQueue = xQueueCreate(1, 0);
    modelReadySemaphore = xSemaphoreCreateMutex();
    
    Serial.println("Inicializando Modelo TFLite...");
    if (voiceModel.initialize(voiceModel_3classes_tflite)) {
        modelReady = true;
        Serial.println("✓ Modelo cargado correctamente");
    } else {
        Serial.println("✗ Error al cargar el modelo");
        while(1) { digitalWrite(LED_BUILTIN, HIGH); delay(100); digitalWrite(LED_BUILTIN, LOW); delay(100); }
    }

    xTaskCreatePinnedToCore(wifiAndAudioTask, "WiFiTask", 8192, NULL, 1, &wifiTaskHandle, 0);
    xTaskCreatePinnedToCore(mlInferenceTask, "MLTask", 16384, NULL, 1, &mlTaskHandle, 1);
}

void loop() {
    // El loop principal está deshabilitado para que FreeRTOS maneje todo
    vTaskDelay(portMAX_DELAY);
}