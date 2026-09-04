#include "read_serial.hpp"
#include <Arduino.h>
#include <vector>

#define HW_SERIAL Serial1

RealtimeFeatureExtractor extractor(WINDOW, WINDOW - STEP, VREF, BITS, WAMP_THRESH);

void init_serial() {
    Serial.begin(115200);
    HW_SERIAL.begin(115200);
}

void read_available_samples() {
    std::vector<float> incoming_batch;
    while (HW_SERIAL.available() >= sizeof(float)) {
        float sample;
        HW_SERIAL.readBytes((char*)&sample, sizeof(float));
        incoming_batch.push_back(sample);
    }

    if (!incoming_batch.empty()) {
        extractor.read_samples(incoming_batch);
    }
}

