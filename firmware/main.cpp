#include <Arduino.h>
#include <vector>
#include <map>
#include <algorithm>
#include <ArduinoEigenDense.h>
#include "read_serial.hpp"
#include "feature_extraction.hpp"
#include "calibration.hpp"
#include "pred.hpp"

#include <SPI.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include "secrets.h"

#define SPIWIFI_SS    10
#define SPIWIFI_ACK    9   // BUSY
#define ESP32_RESETN   8
#define ESP32_GPIO0   -1   // not connected

WiFiUDP udp;
const IPAddress DEST_IP(255, 255, 255, 255);
const uint16_t UDP_PORT = 4210;

const char* WIFI_SSID = "iPhone (27)";

// counted in feature windows (10/s at STEP = 200, 2 kHz), not raw samples. Matches
// pred_realtime.py's 10400 / 5200 raw samples after it drops the first 400
const size_t REST_CALIBRATION_WINDOWS = 49;    // ~5s
const size_t GESTURE_CALIBRATION_WINDOWS = 23; // ~2.6s per active gesture

const int COUNTDOWN_SECONDS = 3;
// windows discarded after a flush. Windowing restarts at the flush, so skipping 2 makes the
// first kept window samples 401-800, same as pred_realtime.py's get_features(raw[400:])
const size_t WINDOWS_TO_SKIP = WINDOW / STEP;
size_t windows_to_skip = 0;

// Status text for display_receiver.py. Packets of 1 byte are gesture ids, anything longer
// is text to show, so every status message must be at least 2 bytes
void send_status(const String& text) {
    udp.beginPacket(DEST_IP, UDP_PORT);
    udp.write(reinterpret_cast<const uint8_t*>(text.c_str()), text.length());
    udp.endPacket();
}

// Prints the prompt (Serial + display), counts down so there's time to get into position,
// then throws away everything sampled during the countdown so calibration only sees the held gesture
void prompt_and_wait(const char* gesture_name) {
    Serial.print("Next: ");
    Serial.println(gesture_name);
    for (int i = COUNTDOWN_SECONDS; i > 0; i--) {
        Serial.print("  ");
        Serial.println(i);
        send_status(String("Next: ") + gesture_name + "\n" + i);
        delay(1000);
    }
    Serial.print("Hold: ");
    Serial.println(gesture_name);
    send_status(String("Hold: ") + gesture_name);
    flush_samples();
    windows_to_skip = WINDOWS_TO_SKIP;
}

const char* class_name_for(int class_id) {
    for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
        if (MODEL_CLASS_IDS[i] == class_id) {
            return MODEL_CLASS_NAMES[i];
        }
    }
    return "unknown";
}

// debug: logs every packet so a silent display can be traced to "never sent" vs
// "AirLift failed" vs "network dropped it"
void send_gesture(uint8_t gesture_id) {
    udp.beginPacket(DEST_IP, UDP_PORT);
    udp.write(&gesture_id, 1);
    int ok = udp.endPacket();
    Serial.print("Sent ");
    Serial.print(class_name_for(gesture_id));
    Serial.println(ok ? "" : "  (endPacket failed)");
}

enum State { CALIBRATE_REST, CALIBRATE_GESTURE, RUNNING };
State state = CALIBRATE_REST;


std::vector<int> ordered_class_ids; 
size_t rest_column = 0;
std::vector<int> active_class_ids;
size_t active_idx = 0;

std::vector<std::vector<float>> rest_feature_samples(NUM_FEATURES);   // [feature][sample]
std::vector<std::vector<float>> gesture_feature_samples(NUM_FEATURES); 

// pred_realtime.py averages LDA scores over the last NUM_WINDOWS_FOR_VOTE windows (~1 s).
// Scores are linear in the features, so averaging the features first picks the same class
const size_t NUM_WINDOWS_FOR_VOTE = 10;
std::vector<std::vector<float>> recent_windows;
size_t recent_head = 0;

std::vector<float> rest_median, rest_mad, rest_mean;
MatrixXd centroids;
lda_model model;

void collect_into(std::vector<std::vector<float>>& dest, const std::vector<float>& values) {
    for (size_t i = 0; i < values.size(); i++) {
        dest[i].push_back(values[i]);
    }
}

void setup() {
    init_serial();
    WiFi.setPins(SPIWIFI_SS, SPIWIFI_ACK, ESP32_RESETN, ESP32_GPIO0, &SPI);
    while (WiFi.begin(WIFI_SSID, WIFI_PASSWORD) != WL_CONNECTED) delay(500);
    Serial.println(WiFi.localIP());
    udp.begin(UDP_PORT);


    for (int i = 0; i < MODEL_NUM_CLASSES; i++) { // ascending id order, matches MODEL_PRIORS
        ordered_class_ids.push_back(MODEL_CLASS_IDS[i]);
        if (i == MODEL_REST_INDEX) {
            rest_column = i;
        } else {
            active_class_ids.push_back(MODEL_CLASS_IDS[i]);
        }
    }
    centroids = MatrixXd(NUM_FEATURES, ordered_class_ids.size());

    // started after the blocking WiFi connect so the ring buffer doesn't overflow during it
    start_sampling();
    prompt_and_wait("rest (relax your hand)");
}

void loop() {
    read_available_samples();
    if (!extractor.features_ready()) {
        return;
    }
    if (windows_to_skip > 0) {
        windows_to_skip--;
        extractor.clear_features_ready();
        return;
    }
    const std::vector<float>& features = extractor.get_features();

    switch (state) {
        case CALIBRATE_REST: {
            collect_into(rest_feature_samples, features);
            if (rest_feature_samples[0].size() >= REST_CALIBRATION_WINDOWS) {
                auto rest_stats = get_rest_stats(rest_feature_samples, NUM_FEATURES, rest_feature_samples[0].size());
                rest_median = rest_stats[0];
                rest_mad = rest_stats[1];
                rest_mean = rest_stats[2];

                for (int i = 0; i < NUM_FEATURES; i++) {
                    centroids(i, rest_column) = rest_mean[i];
                }
                Serial.println("Rest calibration complete.");
                send_status("Rest calibration complete");
                state = CALIBRATE_GESTURE;
                prompt_and_wait(class_name_for(active_class_ids[active_idx]));
            }
            break;
        }
        case CALIBRATE_GESTURE: {
            std::vector<float> calibrated = apply_baseline_calibration(features, rest_median, rest_mad);
            collect_into(gesture_feature_samples, calibrated);
            if (gesture_feature_samples[0].size() >= GESTURE_CALIBRATION_WINDOWS) {
                std::vector<std::vector<std::vector<float>>> single_class = {gesture_feature_samples};
                auto class_centroid = get_live_centroids(single_class, NUM_FEATURES,
                    gesture_feature_samples[0].size(), 1)[0];

                int class_id = active_class_ids[active_idx];
                size_t col = std::find(ordered_class_ids.begin(), ordered_class_ids.end(), class_id)
                    - ordered_class_ids.begin();
                for (int i = 0; i < NUM_FEATURES; i++) {
                    centroids(i, col) = class_centroid[i];
                }

                for (auto& f : gesture_feature_samples) {
                    f.clear();
                }
                active_idx++;

                if (active_idx >= active_class_ids.size()) {
                    load_lda_model(model, centroids);
                    Serial.println("Calibration complete. Predicting.");
                    send_status("Calibration complete");
                    state = RUNNING;
                    // prediction buffer starts fresh, like pred_realtime.py
                    flush_samples();
                    return;
                } else {
                    prompt_and_wait(class_name_for(active_class_ids[active_idx]));
                }
            }
            break;
        }
        case RUNNING: {
            std::vector<float> calibrated = apply_baseline_calibration(features, rest_median, rest_mad);
            if (recent_windows.size() < NUM_WINDOWS_FOR_VOTE) {
                recent_windows.push_back(calibrated);
            } else {
                recent_windows[recent_head] = calibrated;
                recent_head = (recent_head + 1) % NUM_WINDOWS_FOR_VOTE;
            }
            if (recent_windows.size() < NUM_WINDOWS_FOR_VOTE) {
                break;
            }

            VectorXd x = VectorXd::Zero(NUM_FEATURES);
            for (const auto& window : recent_windows) {
                for (int i = 0; i < NUM_FEATURES; i++) {
                    x(i) += window[i]; // explicit float->double copy
                    //  Eigen::Map cant alias a float buffer as vectorxd
                }
            }
            x /= static_cast<double>(recent_windows.size());
            int gesture_id = predict(x, model);
            send_gesture(static_cast<uint8_t>(gesture_id));
            break;
        }
    }

    extractor.clear_features_ready();
}
