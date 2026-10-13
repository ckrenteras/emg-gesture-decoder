#include "read_serial.hpp"
#include <Arduino.h>
#include <vector>

const int EMG_PIN = A0; // pin 14 on the Teensy 4.1
const uint32_t SAMPLE_PERIOD_US = static_cast<uint32_t>(1000000 / SAMPLE_RATE); // 500 us at 2 kHz
const size_t RING_SIZE = 1024; // ~0.5 s of slack if loop() stalls (e.g. WiFi send)

IntervalTimer sample_timer;
volatile uint16_t ring[RING_SIZE];
volatile size_t ring_head = 0; // written only by the ISR
size_t ring_tail = 0;          // read only by loop()

RealtimeFeatureExtractor extractor(WINDOW, WINDOW - STEP, VREF, BITS, WAMP_THRESH);

void sample_isr() {
    ring[ring_head] = analogRead(EMG_PIN);
    ring_head = (ring_head + 1) % RING_SIZE;
}

void init_serial() {
    Serial.begin(115200);
    analogReadResolution(static_cast<unsigned int>(BITS)); // 10-bit, same as training data
}

void start_sampling() {
    sample_timer.begin(sample_isr, SAMPLE_PERIOD_US);
}

void flush_samples() {
    ring_tail = ring_head;
    extractor.reset_windows();
}

// stops as soon as a window completes so loop() handles every window; the rest stays
// in the ring for the next call
void read_available_samples() {
    size_t head = ring_head;
    while (ring_tail != head) {
        float sample = ring[ring_tail];
        ring_tail = (ring_tail + 1) % RING_SIZE;
        if (extractor.push_sample(sample)) {
            return;
        }
    }
}
