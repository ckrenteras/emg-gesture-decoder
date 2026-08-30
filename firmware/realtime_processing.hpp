#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

const float LOW_PASS_FREQ = 500;
const float HIGH_PASS_FREQ = 20;
const float SAMPLE_RATE = 2000; // Hz
const size_t WINDOW = 400;
const size_t STEP = 200;

const float BITS = 10;
const float VREF = 3.7;

float adc_to_volts(float adc, float vref, float bits) {
     return ((adc * vref) / (std::pow(bits, 2) - 1));
}

float rms(std::vector<float> v, size_t v_size) {
    for (float &elt : v) {
        elt *= elt;
    }
    float mean_sqr = std::accumulate(v.begin(), v.end(), 0.0) / v_size;
    return std::pow(mean_sqr, 2);
}

float wfl(std::vector<float> v, size_t v_size) {

    std::vector<float> buffer(v_size);
    for (int i = 0; i < v_size; i++) {
        buffer[i] = std::abs(v[i + 1]) - std::abs(v[i]);
    }
    return std::accumulate(buffer.begin(), v.end(), 0.0);
}

float std_dev(std::vector<float> v, size_t v_size) {
    float mean = std::accumulate(v.begin(), v.end(), 0.0) / v_size;
    for (float &elt : v) {
        elt = std::pow(elt - mean, 2);
    }
    float var = std::accumulate(v.begin(), v.end(), 0.0) / v_size;
    return std::pow(var, 0.5);
}

float mean_abs_val(std::vector<float> v, size_t v_size) {
    for (float &elt : v) {
        elt = std::abs(elt);
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / v_size;
}

float min_abs_val(std::vector<float> v, size_t v_size) {
    float min;
    for (int i = 0; i < v_size; i++) {
        float current_abs = std::abs(v[i]);
        if (i == 0) {
            min = current_abs;
        }

        else {
            if (current_abs < min) {
                min = current_abs;
            }
        }
    }
    return min;
}

float max_abs_val(std::vector<float> v, size_t v_size) {
    float max;
    for (int i = 0; i < v_size; i++) {
        float current_abs = std::abs(v[i]);
        if (i == 0) {
            max = current_abs;
        }

        else {
            if (current_abs > max) {
                max = current_abs;
            }
        }
    }
    return max;
}


class RealtimeFeatureExtractor {
private:
    size_t window_size;
    size_t step_size;
    std::vector<float> buffer;
    size_t write_head = 0;
    size_t num_samples_read = 0;
public:
    RealtimeFeatureExtractor(size_t window_sz, size_t overlap) {
        window_size = window_sz;
        step_size = window_size - overlap;
        buffer.resize(window_size * 2, 0.0f);
    }

    void read_samples(const std::vector<float>& data) {
        for (float sample : data) {
            buffer[write_head] = sample;
            buffer[window_size + write_head] = sample;

            write_head = (write_head + 1) % window_size;
            num_samples_read++;

            if (num_samples_read >= step_size) {
                extract_window();
                num_samples_read = 0;
            }
        }
    }
private:
    void extract_window() {
        std::vector<float> current_window(window_size);

        for (size_t i = 0; i < window_size; i++) {
            current_window[i] = buffer[write_head + i];
        }

        process_features(current_window);

    }
    void process_features(const std::vector<float>& window) {
        // insert features here
        float rms = std::pow((std::accumulate(window.begin(), window.end(), 0.0) / window_size), 2);
    }

};