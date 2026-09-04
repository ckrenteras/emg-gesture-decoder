#include <WiFi.h>
#include <WiFiUdp.h>

// TODO: fill in for network
const char* WIFI_SSID = "YOUR_SSID";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

const uint16_t UDP_PORT = 4210;
const IPAddress DEST_IP(255, 255, 255, 255);

// Matches ESP_SERIAL on the Teensy side in main.cpp
#define TEENSY_SERIAL Serial2
const uint32_t TEENSY_BAUD = 115200;

const uint8_t FRAME_START = 0xAA;
const size_t FRAME_SIZE = 3; // start byte, gesture id, checksum

WiFiUDP udp;

void connect_wifi() {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(300);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("Connected, IP: ");
    Serial.println(WiFi.localIP());
}

// Returns true and fills gesture_id when a valid frame read
// On checksum error drop byte and returns false so the
// next loop() iteration can try to resync on the following start byte
bool read_frame(uint8_t* gesture_id) {
    if (TEENSY_SERIAL.available() < static_cast<int>(FRAME_SIZE)) {
        return false;
    }
    if (TEENSY_SERIAL.peek() != FRAME_START) {
        TEENSY_SERIAL.read();
        return false;
    }

    uint8_t frame[FRAME_SIZE];
    TEENSY_SERIAL.readBytes(frame, FRAME_SIZE);
    uint8_t expected_checksum = frame[0] ^ frame[1];
    if (frame[2] != expected_checksum) {
        return false;
    }

    *gesture_id = frame[1];
    return true;
}

void setup() {
    Serial.begin(115200);
    TEENSY_SERIAL.begin(TEENSY_BAUD);
    connect_wifi();
    udp.begin(UDP_PORT);
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        connect_wifi();
    }

    uint8_t gesture_id;
    if (read_frame(&gesture_id)) {
        udp.beginPacket(DEST_IP, UDP_PORT);
        udp.write(&gesture_id, 1);
        udp.endPacket();
    }
}
