#pragma once
#include <iostream>
#include <vector>

// rest_stats[0] = median, [1] = MAD (scaled like scipy's scale='normal'), [2] = mean
std::vector<std::vector<float>> get_rest_stats(std::vector<std::vector<float>> features, size_t num_features, size_t num_samples);
std::vector<std::vector<float>> get_live_centroids(std::vector<std::vector<std::vector<float>>> feature_by_class,
    size_t num_features, size_t num_samples, size_t num_classes);

std::vector<float> apply_baseline_calibration(const std::vector<float>& features,
    const std::vector<float>& rest_median, const std::vector<float>& rest_mad,
    float clip = 32.0f, float eps = 1e-8f);

// 3rd order Butterworth (one biquad + one first order section), same response as
// scipy's butter(N=3, ...). Run it continuously one sample at a time so the state
// carries across windows, like sosfilt with zi in pred_realtime.py
class ButterworthFilter{
private:
    float b0, b1, b2, a1, a2; // biquad coeffs
    float x1, x2, y1, y2;     // biquad history
    float c0, c1, d1;         // first order coeffs
    float u1, w1;             // first order history

public:
    //init states
    ButterworthFilter();

    enum FilterType { LOWPASS, HIGHPASS };

    void configure(FilterType type, float cutoff_freq, float sample_rate);
    float process(float x0);

    void reset();
};
