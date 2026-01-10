#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"

// --- Configuration ---
#define potentionmeterPin 35
#define microphonePin 34

#define SAMPLE_RATE 10000
#define RECORD_TIME 2 // segundos
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_TIME)

// Reservamos el buffer en la memoria del ESP32
uint16_t audioBuffer[TOTAL_SAMPLES]; 
bool isRecording = false;

char bedroomLightIP[16] = "";
const int wizPort = 38899;

WiFiUDP udp;
int temp;
int brightness;
int potValue;


// Function to discover the light IP by its MAC address
bool findLightIP(const char* targetMAC = bedroom_light_mac, char* targetIP = bedroomLightIP) {
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
                    // COPIA CORRECTA: Copiamos el contenido a nuestro buffer seguro
                    strncpy(targetIP, udp.remoteIP().toString().c_str(), 15);
                    targetIP[15] = '\0'; // Aseguramos el fin de cadena
                    Serial.printf("Found Light! IP: %s\n", targetIP);
                    return true;
                }
            }
        }
    }
    return false;
}

// Function to send command to Wiz Light
void sendWizCommand(const char* jsonCommand, char* targetIP=bedroomLightIP) {
    if (targetIP[0] == '\0'){ // Don't send if we don't have an IP
        while (!findLightIP()) {
            delay(3000); 
        }
    }; 
    udp.beginPacket(targetIP, wizPort);
    udp.print(jsonCommand);
    udp.endPacket();
    Serial.print("Sent: ");
    Serial.println(jsonCommand);
}

void turnOn() {
    // Standard ON command used by pywizlight
    const char* cmd = "{\"method\":\"setPilot\",\"params\":{\"state\":true}}";
    sendWizCommand(cmd);
}

void turnOff() {
    // Standard OFF command
    const char* cmd = "{\"method\":\"setPilot\",\"params\":{\"state\":false}}";
    sendWizCommand(cmd);
}

/**
 * @param brightness: 10 to 100 (percentage)
 * @param kelvin: 2700 to 6500 (standard range for most WiZ bulbs)
 */
void setWizLight(int brightness, int kelvin, int sceneID = 6, char* targetIP=bedroomLightIP) {
    if (targetIP[0] == '\0'){ // Don't send if we don't have an IP
        while (!findLightIP()) {
            delay(3000); 
        }
    }; 
    // Constrain values to prevent errors
    brightness = constrain(brightness, 10, 100);
    kelvin = constrain(kelvin, 2700, 6500);

    // Create JSON document
    JsonDocument doc;
    doc["method"] = "setPilot";
    
    JsonObject params = doc["params"].to<JsonObject>();
    params["state"] = true;
    params["dimming"] = brightness;
    params["temp"] = kelvin;
    params["sceneID"] = sceneID;

    // Serialize JSON to a string
    char buffer[128];
    serializeJson(doc, buffer);

    // Send via UDP
    udp.beginPacket(targetIP, wizPort);
    udp.print(buffer);
    udp.endPacket();

    Serial.printf("Sent: Brightness %d%%, Temp %dK\n", brightness, kelvin);
}

void Pot2Light() {
    int off_threshold = 120;
    int minTemp_threshold = 600;

    int newPotValue = analogRead(potentionmeterPin);
    if (abs(newPotValue - potValue) < 82) {
        return; // Avoid small changes, 2% 4095 ~ 82 
    }
    potValue = newPotValue;

    if (potValue < off_threshold) {
        turnOff();
        brightness = 0;
        return;
    } else if (potValue < minTemp_threshold) {
        setWizLight(10, 2700);
        return;
    }

    brightness = map(potValue, minTemp_threshold, 4095, 10, 100); // Map to 10-100%
    temp = map(potValue, minTemp_threshold, 4095, 2700, 6500); // Map to 2700-6500K 
    setWizLight(brightness, temp);
}

void recordAudio() {
    Serial.println(">>> Iniciando grabación (2 seg)...");
    
    // Calculamos el tiempo entre muestras en microsegundos
    unsigned int sampleDelay = 1000000 / SAMPLE_RATE;
    unsigned long startTime = millis();

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        unsigned long nextSampleTime = micros() + sampleDelay;
        
        audioBuffer[i] = analogRead(microphonePin);
        
        // Esperamos con precisión de microsegundos para mantener la frecuencia
        while (micros() < nextSampleTime) {
            // Espera activa para precisión
        }
    }

    Serial.printf(">>> Grabación finalizada. Tiempo total: %lu ms\n", millis() - startTime);
}

// Función para enviar los datos al Serial y graficarlos (Serial Plotter)
void playBackSerial() {
    Serial.println("Enviando datos de audio al monitor...");
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        Serial.println(audioBuffer[i]);
        // delayMicroseconds(100); // Opcional, para no saturar el buffer del PC
    }
}


void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(potentionmeterPin, INPUT);
    pinMode(microphonePin, INPUT);
    analogReadResolution(12); // Máxima resolución del ADC
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    
    udp.begin(wizPort); // Start UDP
    delay(500);

    // Keep trying to find the light before starting the main loop
    while (!findLightIP()) {
        delay(3000); 
    }
    //digitalWrite(LED_BUILTIN, HIGH);

    Serial.println("Presiona 'g' en el monitor serial para grabar.");
    
}

void loop() {
    //Pot2Light();
    //delay(200);

    if (Serial.available() > 0) {
        char c = Serial.read();
        if (c == 'g') {
            digitalWrite(LED_BUILTIN, HIGH);
            recordAudio();
            digitalWrite(LED_BUILTIN, LOW);
            playBackSerial();
        }
    }
}