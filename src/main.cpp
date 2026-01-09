#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"

// --- Configuration ---
#define potentionmeterPin 35
#define microphonePin 34

const char* wizIP = "192.168.1.2"; // Replace with your light's IP
const int wizPort = 38899;

WiFiUDP udp;
int temp;
int brightness;

// Function to send command to Wiz Light
void sendWizCommand(const char* jsonCommand) {
    udp.beginPacket(wizIP, wizPort);
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
void setWizLight(int brightness, int kelvin, int sceneID = 6) {
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
    udp.beginPacket(wizIP, wizPort);
    udp.print(buffer);
    udp.endPacket();

    Serial.printf("Sent: Brightness %d%%, Temp %dK\n", brightness, kelvin);
}

void Pot2Light() {
    int off_threshold = 80;
    int minTemp_threshold = 500;

    int potValue = analogRead(potentionmeterPin);
    if (potValue < off_threshold) {
        turnOff();
        brightness = 0;
        return;
    } else if (potValue < minTemp_threshold) {
        setWizLight(10, 2700);
    }

    int new_brightness = map(potValue, off_threshold, 4095, 10, 100); // Map to 10-100%
    if (abs(new_brightness - brightness) < 3) {
        return; // Avoid small changes
    }
    brightness = new_brightness;
    temp = map(potValue, off_threshold, 4095, 2700, 6500); // Map to 2700-6500K 
    setWizLight(brightness, temp);
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(potentionmeterPin, INPUT);
    pinMode(microphonePin, INPUT);
    
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
}

void loop() {
    Pot2Light();
    delay(200);
}