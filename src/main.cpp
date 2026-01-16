#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"
#include "audio_processing.h"
#include "mfcc.h"
#include "tflite_inference.h"
#include "model_data.h"  // Embedded TFLite model
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"

// --- Configuration ---
#define MICROPHONE_PIN 34
#define POTENTIOMETER_PIN 35

#define SAMPLE_RATE 12000
#define RECORD_TIME 2
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_TIME)

// --- Audio Buffers ---
// Double buffering for audio capture (one for recording, one for processing)
float audioBuffer1[TOTAL_SAMPLES];
float audioBuffer2[TOTAL_SAMPLES];
float* currentRecordingBuffer = audioBuffer1;
float* processedAudioBuffer = nullptr;

// --- Processing Buffers ---
float processed_audio[TOTAL_SAMPLES];
float mfcc_features[N_MFCC * MAX_TIME_STEPS];  // (13 x 64)
float mfcc_normalized[N_MFCC * MAX_TIME_STEPS];

// --- Global Objects ---
AudioProcessor audioProcessor;
MFCCExtractor mfccExtractor;
VoiceModelInference voiceModel;

// --- WiFi and Control ---
WiFiUDP udp;
char bedroomLightIP[16] = "";
const int wizPort = 38899;
int currentBrightness = 50;
int currentTemp = 4100;

// --- FreeRTOS Queues and Synchronization ---
QueueHandle_t audioReadyQueue;  // Signal when audio is ready for processing
SemaphoreHandle_t modelReadySemaphore;  // Protect model prediction access

// --- Task Handles ---
TaskHandle_t wifiTaskHandle = nullptr;
TaskHandle_t mlTaskHandle = nullptr;

bool isRecording = false;
bool modelReady = false;
volatile int lastPrediction = -1;
volatile float lastConfidence = 0.0f;

// --- Forward Declarations ---
void wifiAndAudioTask(void* pvParameters);
void mlInferenceTask(void* pvParameters);
bool findLightIP(const char* targetMAC, char* targetIP);
void sendWizCommand(const char* jsonCommand, char* targetIP);
void turnOn();
void turnOff();
void setWizLight(int brightness, int kelvin, int sceneID, char* targetIP);
void Pot2Light();
void recordAudio();
void playBackSerial();
void handlePrediction(int classId, float confidence);

// ============================================================================
// WiFi and Light Control Functions (unchanged from original)
// ============================================================================

bool findLightIP(const char* targetMAC, char* targetIP) {
    Serial.println("Searching for WiZ light...");
    
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
                    Serial.printf("Found Light! IP: %s\n", targetIP);
                    return true;
                }
            }
        }
    }
    return false;
}

void sendWizCommand(const char* jsonCommand, char* targetIP) {
    if (targetIP[0] == '\0') {
        while (!findLightIP(bedroom_light_mac, targetIP)) {
            vTaskDelay(3000 / portTICK_PERIOD_MS);
        }
    }
    udp.beginPacket(targetIP, wizPort);
    udp.print(jsonCommand);
    udp.endPacket();
    Serial.print("Sent: ");
    Serial.println(jsonCommand);
}

void turnOn() {
    const char* cmd = "{\"method\":\"setPilot\",\"params\":{\"state\":true}}";
    sendWizCommand(cmd, bedroomLightIP);
}

void turnOff() {
    const char* cmd = "{\"method\":\"setPilot\",\"params\":{\"state\":false}}";
    sendWizCommand(cmd, bedroomLightIP);
}

void setWizLight(int brightness, int kelvin, int sceneID, char* targetIP) {
    if (targetIP[0] == '\0') {
        while (!findLightIP(bedroom_light_mac, targetIP)) {
            vTaskDelay(3000 / portTICK_PERIOD_MS);
        }
    }
    
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

    udp.beginPacket(targetIP, wizPort);
    udp.print(buffer);
    udp.endPacket();

    Serial.printf("Sent: Brightness %d%%, Temp %dK\n", brightness, kelvin);
}

void Pot2Light() {
    int off_threshold = 120;
    int minTemp_threshold = 600;
    static int lastPotValue = 0;

    int newPotValue = analogRead(POTENTIOMETER_PIN);
    if (abs(newPotValue - lastPotValue) < 82) {
        return;
    }
    lastPotValue = newPotValue;

    if (newPotValue < off_threshold) {
        turnOff();
        currentBrightness = 0;
        return;
    } else if (newPotValue < minTemp_threshold) {
        setWizLight(10, 2700, 6, bedroomLightIP);
        currentBrightness = 10;
        currentTemp = 2700;
        return;
    }

    currentBrightness = map(newPotValue, minTemp_threshold, 4095, 10, 100);
    currentTemp = map(newPotValue, minTemp_threshold, 4095, 2700, 6500);
    setWizLight(currentBrightness, currentTemp, 6, bedroomLightIP);
}

// ============================================================================
// Audio Recording Functions
// ============================================================================

void recordAudio() {
    Serial.println(">>> Starting audio recording (2 sec)...");
    digitalWrite(LED_BUILTIN, HIGH);
    
    unsigned int sampleDelay = 1000000 / SAMPLE_RATE;
    unsigned long startTime = millis();

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long nextSampleTime = micros() + sampleDelay;
        
        // Read from ADC (12-bit: 0-4095)
        uint16_t adcValue = analogRead(MICROPHONE_PIN);
        // Convert to float [-1, 1]
        currentRecordingBuffer[i] = (adcValue / 2048.0f) - 1.0f;
        
        // Wait for next sample time
        while (micros() < nextSampleTime) {
            // Busy wait for precision
        }
    }

    Serial.printf(">>> Recording finished. Time: %lu ms\n", millis() - startTime);
    digitalWrite(LED_BUILTIN, LOW);
}

void playBackSerial() {
    Serial.println("Sending audio data to monitor...");
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        Serial.println(currentRecordingBuffer[i]);
    }
}

// ============================================================================
// Voice Recognition and Light Control
// ============================================================================

void handlePrediction(int classId, float confidence) {
    Serial.printf("Prediction: Class=%d, Confidence=%.2f%%\n", classId, confidence * 100);
    Serial.printf("Class Name: %s\n", voiceModel.get_class_name(classId));
    
    // Store last prediction
    lastPrediction = classId;
    lastConfidence = confidence;
    
    // Only act if confidence is high enough
    if (confidence < 0.6f) {
        Serial.println("Confidence too low, ignoring prediction");
        return;
    }
    
    // Execute action based on prediction
    switch (classId) {
        case CLASS_AMBIENTE:
            //Serial.println("Setting ambient lighting...");
            //setWizLight(30, 4000, 6, bedroomLightIP);
            break;
            
        case CLASS_APAGAR_LUZ:
            Serial.println("Turning off light...");
            turnOff();
            break;
            
        case CLASS_PRENDER_LUZ:
            Serial.println("Turning on light...");
            turnOn();
            break;
            
        default:
            Serial.println("Unknown class");
            break;
    }
}

// ============================================================================
// FreeRTOS Task: WiFi and Audio (Core 0)
// ============================================================================

void wifiAndAudioTask(void* pvParameters) {
    Serial.println("WiFi & Audio Task started on Core 0");
    
    // Initialize WiFi
    WiFi.begin(ssid, password);
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifiAttempts < 20) {
        vTaskDelay(500 / portTICK_PERIOD_MS);
        Serial.print(".");
        wifiAttempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi Connected!");
        digitalWrite(LED_BUILTIN, HIGH);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
        digitalWrite(LED_BUILTIN, LOW);
    }
    
    // Initialize UDP
    udp.begin(wizPort);
    
    // Find light IP
    while (!findLightIP(bedroom_light_mac, bedroomLightIP)) {
        vTaskDelay(3000 / portTICK_PERIOD_MS);
    }
    
    // Main loop
    while (1) {
        // Check for serial commands
        if (Serial.available() > 0) {
            char c = Serial.read();
            if (c == 'g') {
                // Record audio
                recordAudio();
                
                // Signal ML task that audio is ready
                xQueueSend(audioReadyQueue, nullptr, portMAX_DELAY);
                
                // Optionally send to serial for debugging
                // playBackSerial();
            }
        }
        
        // Update potentiometer-based light control
        Pot2Light();
        
        vTaskDelay(200 / portTICK_PERIOD_MS);
    }
}

// ============================================================================
// FreeRTOS Task: ML Inference (Core 1)
// ============================================================================

void mlInferenceTask(void* pvParameters) {
    Serial.println("ML Inference Task started on Core 1");
    
    // Wait until model is ready
    while (!modelReady) {
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    
    Serial.println("Model ready, starting inference loop...");
    
    while (1) {
        // Wait for audio to be ready (blocks until audio is received)
        if (xQueueReceive(audioReadyQueue, nullptr, portMAX_DELAY)) {
            unsigned long startTime = millis();
            
            Serial.println("\n=== Processing Audio ===");
            
            // 1. Preprocess audio
            Serial.println("1. Preprocessing audio...");
            audioProcessor.process_complete_pipeline(
                currentRecordingBuffer, TOTAL_SAMPLES, processed_audio);
            
            // 2. Extract MFCC features
            Serial.println("2. Extracting MFCC features...");
            int num_frames = mfccExtractor.extract_mfcc(
                processed_audio, TOTAL_SAMPLES, mfcc_features, MAX_TIME_STEPS);
            
            Serial.printf("   Extracted %d frames\n", num_frames);
            
            // 3. Pad/truncate to fixed length
            Serial.println("3. Padding to fixed length...");
            mfccExtractor.pad_to_fixed_length(
                mfcc_features, num_frames, mfcc_normalized, MAX_TIME_STEPS);
            
            // 4. Normalize MFCC
            //Serial.println("4. Normalizing MFCC...");
            //mfccExtractor.normalize_mfcc(
            //    mfcc_normalized, N_MFCC, MAX_TIME_STEPS);
            
            // 5. Run inference
            Serial.println("5. Running inference...");
            int classId = voiceModel.predict(mfcc_features); //.predict(mfcc_normalized)
            
            // Get confidence
            float probabilities[NUM_CLASSES];
            voiceModel.get_output_probabilities(probabilities);
            float confidence = (classId >= 0) ? probabilities[classId] : 0.0f;
            
            unsigned long processingTime = millis() - startTime;
            Serial.printf("\nProcessing time: %lu ms\n", processingTime);
            
            // 6. Handle prediction result (with synchronization)
            xSemaphoreTake(modelReadySemaphore, portMAX_DELAY);
            handlePrediction(classId, confidence);
            xSemaphoreGive(modelReadySemaphore);
        }
    }
}

// ============================================================================
// Setup and Main
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    // Initialize GPIO
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(POTENTIOMETER_PIN, INPUT);
    pinMode(MICROPHONE_PIN, INPUT);
    analogReadResolution(12);
    
    Serial.println("\n=== Voice Recognition System Starting ===");
    
    // Create FreeRTOS queues and semaphores
    audioReadyQueue = xQueueCreate(1, 0);  // Binary queue
    modelReadySemaphore = xSemaphoreCreateMutex();
    
    // Initialize TFLite model
    Serial.println("Initializing TFLite model...");
    if (voiceModel.initialize(voiceModel_3classes_tflite)) {
        modelReady = true;
        Serial.println("✓ Model initialized successfully");
    } else {
        Serial.println("✗ Failed to initialize model");
        while (1) {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
            delay(500);
        }
    }
    
    // Create FreeRTOS tasks
    // Task 1: WiFi + Audio (Core 0)
    xTaskCreatePinnedToCore(
        wifiAndAudioTask,      // Task function
        "WiFi_Audio_Task",     // Task name
        8192,                  // Stack size (8KB)
        nullptr,               // Parameter
        1,                     // Priority
        &wifiTaskHandle,       // Task handle
        0                      // Core ID (0)
    );
    
    // Task 2: ML Inference (Core 1)
    xTaskCreatePinnedToCore(
        mlInferenceTask,       // Task function
        "ML_Inference_Task",   // Task name
        16384,                 // Stack size (16KB) - larger for ML
        nullptr,               // Parameter
        1,                     // Priority
        &mlTaskHandle,         // Task handle
        1                      // Core ID (1)
    );
    
    Serial.println("\n=== FreeRTOS Tasks Created ===");
    Serial.println("Core 0: WiFi + Audio Recording");
    Serial.println("Core 1: ML Inference");
    Serial.println("\nSend 'g' via serial to record audio and run inference");
}

void loop() {
    // Main loop is handled by FreeRTOS scheduler
    // This function will not be called frequently
    vTaskDelay(1000 / portTICK_PERIOD_MS);
}