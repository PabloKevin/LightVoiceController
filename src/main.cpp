#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include "credentials.h"

// --- Configuration ---
const char* wizIP = "192.168.1.2"; // Replace with your light's IP
const int wizPort = 38899;

WiFiUDP udp;

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

void setup() {
    Serial.begin(115200);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");
    
    udp.begin(wizPort); // Start UDP
}

void loop() {
    // Example: Toggle every 5 seconds
    turnOn();
    delay(5000);
    turnOff();
    delay(5000);
}