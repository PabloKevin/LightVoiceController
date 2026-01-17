#include "wifi_control.h"
#include <WiFi.h>

static const char* ssid = "YOUR_SSID";
static const char* pass = "YOUR_PASS";
static const char* host = "192.168.1.50";

void wifi_init() {
    WiFi.begin(ssid, pass);
    while (WiFi.status() != WL_CONNECTED) delay(500);
}

void send_cmd(const char* cmd) {
    WiFiClient client;
    if (client.connect(host, 80)) {
        client.printf("GET /%s HTTP/1.1\r\nHost: %s\r\n\r\n", cmd, host);
        client.stop();
    }
}

void light_on() { send_cmd("on"); }
void light_off() { send_cmd("off"); }
