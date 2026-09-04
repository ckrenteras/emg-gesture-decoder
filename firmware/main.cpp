#include <Arduino.h>
#include <vector>
#include <map>
#include <algorithm>
#include <Eigen/Dense>
#include "read_serial.hpp"
#include "feature_extraction.hpp"
#include "calibration.hpp"
#include "pred.hpp"

#define ESP_SERIAL Serial2
const uint8_t FRAME_START = 0xAA;

void send_gesture(uint8_t gesture_id) {
    uint8_t frame[3] = {FRAME_START, gesture_id, static_cast<uint8_t>(FRAME_START ^ gesture_id)};
    ESP_SERIAL.write(frame, sizeof(frame));
}

const size_t REST_CALIBRATION_SAMPLES = 10400;   // ~5s 
const size_t GESTURE_CALIBRATION_SAMPLES = 5200; // ~2.6s per active gesture

enum State { CALIBRATE_REST, CALIBRATE_GESTURE, RUNNING };
State state = CALIBRATE_REST;


std::map<int, std::string> class_mapping;
std::vector<int> ordered_class_ids; 
size_t rest_column = 0;
std::vector<int> active_class_ids;
size_t active_idx = 0;

std::vector<std::vector<float>> rest_feature_samples(NUM_FEATURES);   // [feature][sample]
std::vector<std::vector<float>> gesture_feature_samples(NUM_FEATURES); 

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
    ESP_SERIAL.begin(115200);

    class_mapping = load_class_mappings(MAPPINGS_PATH);
    for (const auto& [id, name] : class_mapping) { // std::map iterates in ascending key order
        ordered_class_ids.push_back(id);
        if (name == "rest") {
            rest_column = ordered_class_ids.size() - 1;
        } else {
            active_class_ids.push_back(id);
        }
    }
    centroids = MatrixXd(NUM_FEATURES, ordered_class_ids.size());
}

void loop() {
    read_available_samples();
    if (!extractor.features_ready()) {
        return;
    }
    const std::vector<float>& features = extractor.get_features();

    switch (state) {
        case CALIBRATE_REST: {
            collect_into(rest_feature_samples, features);
            if (rest_feature_samples[0].size() >= REST_CALIBRATION_SAMPLES) {
                auto rest_stats = get_rest_stats(rest_feature_samples, NUM_FEATURES, rest_feature_samples[0].size());
                rest_median = rest_stats[0];
                rest_mad = rest_stats[1];
                rest_mean = rest_stats[2];

                for (int i = 0; i < NUM_FEATURES; i++) {
                    centroids(i, rest_column) = rest_mean[i];
                }
                state = CALIBRATE_GESTURE;
            }
            break;
        }
        case CALIBRATE_GESTURE: {
            std::vector<float> calibrated = apply_baseline_calibration(features, rest_median, rest_mad);
            collect_into(gesture_feature_samples, calibrated);
            if (gesture_feature_samples[0].size() >= GESTURE_CALIBRATION_SAMPLES) {
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
                    load_lda_model(model, PRIORS_PATH, centroids, MAPPINGS_PATH, INV_COV_PATH,
                        NUM_FEATURES, NUM_FEATURES);
                    state = RUNNING;
                }
            }
            break;
        }
        case RUNNING: {
            std::vector<float> calibrated = apply_baseline_calibration(features, rest_median, rest_mad);
            VectorXd x(calibrated.size());
            for (size_t i = 0; i < calibrated.size(); i++) {
                x(i) = calibrated[i]; // explicit float->double copy
                //  Eigen::Map cant alias a float buffer as vectorxd
            }
            int gesture_id = predict(x, model);
            send_gesture(static_cast<uint8_t>(gesture_id));
            break;
        }
    }

    extractor.clear_features_ready();
}
