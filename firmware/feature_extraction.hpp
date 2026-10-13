#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include "calibration.hpp"

const float LOW_PASS_FREQ = 500;
const float HIGH_PASS_FREQ = 20;
const float SAMPLE_RATE = 2000; // Hz
const size_t WINDOW = 400;
const size_t STEP = 200;
const size_t NUM_FREQ_BINS = WINDOW / 2 + 1; // np.fft.rfft of a 400 sample window
const float WAMP_THRESH = 0.003;
const int NUM_FEATURES = 10; // must equal MODEL_NUM_FEATURES (static_assert in pred.cpp)

const float BITS = 10;
const float VREF = 3.7;


// features
float adc_to_volts(float adc, float vref, float bits);

float rms(std::vector<float> v, size_t v_size);
float wfl(std::vector<float> v, size_t v_size);
float std_dev(std::vector<float> v, size_t v_size);
float mean_abs_val(std::vector<float> v, size_t v_size);
float min_abs_val(std::vector<float> v, size_t v_size);
float max_abs_val(std::vector<float> v, size_t v_size);

float zc(std::vector<float> v, size_t v_size, float prev_sample);

std::vector<float> power_spectrum(const std::vector<float>& v, size_t v_size,
    const std::vector<float>& cos_table, const std::vector<float>& sin_table);

float mean_freq(std::vector<float> freqs, std::vector<float> power, size_t num_bins);

float median_freq(std::vector<float> freqs, std::vector<float> power, size_t num_bins);

float ssc(std::vector<float> v, size_t v_size);

float wamp(std::vector<float> v, size_t v_size, float wamp_thresh);

class RealtimeFeatureExtractor {
private:
    size_t window_size;
    size_t step_size;
    size_t history_len; // window + the one sample before it (for zero crossings)
    std::vector<float> buffer;
    size_t write_head = 0;
    float vref;
    float bits;
    float thresh;
    std::vector<float> features;
    size_t total_samples_seen = 0;
    std::vector<float> freqs;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    ButterworthFilter lowpass;
    ButterworthFilter highpass;
    bool has_new_features = false;
public:
    RealtimeFeatureExtractor(size_t window_sz, size_t overlap, float ref_v, float num_bits, float wamp_threshold);
    const std::vector<float>& get_features() const;
    bool features_ready() const;
    // converts + filters one raw ADC sample; returns true when it completed a window
    bool push_sample(float adc);
    void clear_features_ready();
    // restart windowing from scratch (filter state is kept, like pred_realtime.py)
    void reset_windows();
private:
    void extract_window();
    void process_features(const std::vector<float>& window, float prev_sample);
};
