#include <iostream>
#include "calibration.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// matches scipy.stats.median_abs_deviation(scale='normal'), which pred_realtime.py uses
const float MAD_NORMAL_SCALE = 1.482602218505602f;

// numpy/pandas style median, averages the middle two for an even count
static float median_of(std::vector<float> v) {
    size_t n = v.size();
    auto mid = v.begin() + n / 2;
    std::nth_element(v.begin(), mid, v.end());
    float median = *mid;
    if (n % 2 == 0) {
        median = (*std::max_element(v.begin(), mid) + median) / 2.0f;
    }
    return median;
}

std::vector<std::vector<float>> get_rest_stats(std::vector<std::vector<float>> features, size_t num_features, size_t num_samples) {
    std::vector<std::vector<float>> rest_stats(3, std::vector<float>(num_features));
    for (size_t i = 0; i < num_features; i++) {
        const std::vector<float>& feature = features[i];
        float median = median_of(feature);
        rest_stats[0][i] = median;

        std::vector<float> devs(num_samples);
        std::transform(feature.begin(), feature.end(), devs.begin(), [median](float x) {
            return std::abs(x - median);
        });
        rest_stats[1][i] = median_of(devs) * MAD_NORMAL_SCALE;

        float mean = std::accumulate(feature.begin(), feature.end(), 0.0f) / num_samples;
        rest_stats[2][i] = mean;
    }

    return rest_stats;
}

std::vector<std::vector<float>> get_live_centroids(std::vector<std::vector<std::vector<float>>> feature_by_class, size_t num_features, 
    size_t num_samples, size_t num_classes) {
    std::vector<std::vector<float>> centroids(num_classes, std::vector<float>(num_features));
    for (size_t i = 0; i < num_classes; i++) {
        for (size_t j = 0; j < num_features; j++) {
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

// pass-through until configure() is called
ButterworthFilter::ButterworthFilter()
    : b0(1), b1(0), b2(0), a1(0), a2(0), c0(1), c1(0), d1(0) {
    reset();
}

void ButterworthFilter::configure(FilterType type, float cutoff_freq, float sample_rate) {
    // analog prototype (s + 1)(s^2 + s + 1), bilinear transform with prewarping
    const float pi = 3.1415926535f;
    float K = std::tan((pi * cutoff_freq) / sample_rate);
    float K2 = K * K;

    const float Q = 1.0f; // the 2nd order factor of a 3rd order Butterworth

    float norm = 1.0f / (1.0f + (K / Q) + K2);

    if (type == LOWPASS) {
        b0 = K2 * norm;
        b1 = 2.0f * b0;
        b2 = b0;
    }
    else {
        b0 = norm;
        b1 = -2.0f * b0;
        b2 = b0;
    }

    a1 = 2.0f * (K2 - 1.0f) * norm;
    a2 = (1.0f - (K / Q) + K2) * norm;

    float norm1 = 1.0f / (1.0f + K);
    if (type == LOWPASS) {
        c0 = K * norm1;
        c1 = c0;
    }
    else {
        c0 = norm1;
        c1 = -norm1;
    }
    d1 = (K - 1.0f) * norm1;
}

float ButterworthFilter::process(float x0) {
    float y0 = (b0 * x0) + (b1 * x1) - (a1 * y1) + (b2 * x2) - (a2*y2);

    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;

    float w0 = (c0 * y0) + (c1 * u1) - (d1 * w1);
    u1 = y0;
    w1 = w0;

    return w0;
}

void ButterworthFilter::reset() {
    x1 = 0;
    x2 = 0;
    y1 = 0;
    y2 = 0;
    u1 = 0;
    w1 = 0;
}
