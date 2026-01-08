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

/**
 * @param brightness: 10 to 100 (percentage)
 * @param kelvin: 2700 to 6500 (standard range for most WiZ bulbs)
 */
void setWizLight(int brightness, int kelvin) {
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

    // Serialize JSON to a string
    char buffer[128];
    serializeJson(doc, buffer);

    // Send via UDP
    udp.beginPacket(wizIP, wizPort);
    udp.print(buffer);
    udp.endPacket();

    Serial.printf("Sent: Brightness %d%%, Temp %dK\n", brightness, kelvin);
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected!");
    digitalWrite(LED_BUILTIN, HIGH);
    
    udp.begin(wizPort); // Start UDP
}

bool ramp = false;
int temp = 4000;
bool on = true;

void loop() {
    if (Serial.available() > 0) {
        String command = "";
        command = Serial.readStringUntil('\n');
        switch (command.charAt(0)) {
            case '0': turnOff(); on=false; break;
            case '1': turnOn(); on=true; break;
            case 'W': temp = 4000; break;
            case 'Y': temp = 6500; break;
            case 'R': temp = 2700; break;
            case 's': ramp = true; break;
            case 'f': ramp = false; break;
            default: Serial.println("Unknown command"); return;
        }
        if (ramp){
            for (int b = 10; b <= 100; b += 5) {
                setWizLight(b, temp); // 4000K is a neutral white
                delay(500); 
            }
        } else if (on){
            setWizLight(90, temp);
        }
        delay(1000);
    }
}