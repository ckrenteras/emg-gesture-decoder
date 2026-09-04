#include <iostream>
#include "calibration.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

std::vector<std::vector<float>> get_rest_stats(std::vector<std::vector<float>> features, size_t num_features, size_t num_samples) {
    std::vector<std::vector<float>> rest_stats(3, std::vector<float>(num_features));
    std::vector<float> feature(num_samples);
    for (int i = 0; i < num_features; i++) {
        feature = features[i];
        auto mid = feature.begin() + num_samples / 2;
        std::nth_element(feature.begin(), mid, feature.end());
        float median = *mid;
        rest_stats[0][i] = median;

        std::vector<float> devs(num_samples);
        std::transform(feature.begin(), feature.end(), devs.begin(), [median](float x) {
            return std::abs(x - median);
        });

        auto dev_mid = devs.begin() + num_samples / 2;
        std::nth_element(devs.begin(), dev_mid, devs.end());
        double mad = *dev_mid;
        if (num_samples % 2 == 0) {
            auto max_dev_left = std::max_element(devs.begin(), dev_mid);
            mad = (*max_dev_left + mad) / 2.0f;
        }
        rest_stats[1][i] = mad;

        float mean = std::accumulate(feature.begin(), feature.end(), 0.0f) / num_samples;
        rest_stats[2][i] = mean;
    }

    return rest_stats;
}

std::vector<std::vector<float>> get_live_centroids(std::vector<std::vector<std::vector<float>>> feature_by_class, size_t num_features, 
    size_t num_samples, size_t num_classes) {
    std::vector<std::vector<float>> centroids(num_classes, std::vector<float>(num_features));
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            centroids[i][j] = std::accumulate(feature_by_class[i][j].begin(), feature_by_class[i][j].end(), 0.0f) / num_samples;
        }
    }

    return centroids;
}

std::vector<float> apply_baseline_calibration(const std::vector<float>& features,
    const std::vector<float>& rest_median, const std::vector<float>& rest_mad,
    float clip, float eps) {
    std::vector<float> calibrated(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        float z = (features[i] - rest_median[i]) / (rest_mad[i] + eps);
        calibrated[i] = std::max(-clip, std::min(clip, z));
    }
    return calibrated;
}

void ButterworthFilter::configure(FilterType type, float cutoff_freq, float sample_rate) {
    const float pi = 3.1415926535f;
    float K = std::tan((pi * cutoff_freq) / sample_rate);
    float K2 = K * K;

    const float Q = 0.7071067811f; // fixed for 2nd order

    float norm = 1.0f / (1.0f + (K / Q) + K2);

    if (type == LOWPASS) {
        b0 = K2 * norm;
        b1 = 2.0 * b0;
        b2 = b0;
    }
    else {
        b0 = norm;
        b1 = -2.0f* b0;
        b2 = b0;
    }

    a1 = 2.0f * (K2 - 1.0f) * norm;
    a2 = (1.0f - (K / Q) + K2) * norm;
}
    
float ButterworthFilter::process(float x0) {
    float y0 = (b0 * x0) + (b1 * x1) - (a1 * y1) + (b2 * x2) - (a2*y2);

    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
        
    return y0;
}

void ButterworthFilter::reset() {
    x1 = 0;
    x2 = 0;
    y1 = 0;
    y2 = 0;
}
