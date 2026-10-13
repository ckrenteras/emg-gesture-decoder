#include "feature_extraction.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>


float adc_to_volts(float adc, float vref, float bits) {
    return (adc * vref) / (std::pow(2, bits) - 1);
}

float rms(std::vector<float> v, size_t v_size) {
    float mean_sqr = 0.0;
    for (size_t i = 0; i < v_size; i++) {
        mean_sqr += std::pow(v[i], 2);
    } 
    mean_sqr /= v_size;
    return std::pow(mean_sqr, 0.5);
}

float wfl(std::vector<float> v, size_t v_size) {
    float len = 0.0;
    for (size_t i = 0; i < v_size - 1; i++) {
        len += std::abs(v[i + 1] - v[i]);
    }
    return len;
}

float std_dev(std::vector<float> v, size_t v_size) {
    float mean = std::accumulate(v.begin(), v.end(), 0.0) / v_size;
    float var = 0.0;
    for (size_t i = 0; i < v_size; i++) {
        var += std::pow(v[i] - mean, 2);
    }
    var /= (v_size - 1); // ddof=1, same as pandas rolling().std()
    return std::pow(var, 0.5);
}

float mean_abs_val(std::vector<float> v, size_t v_size) {
   float mean_abs = 0.0;
    for (size_t i = 0; i < v_size; i++) {
        mean_abs += std::abs(v[i]);
    }
    return mean_abs /= v_size;
}

float min_abs_val(std::vector<float> v, size_t v_size) {
    float min;
    for (size_t i = 0; i < v_size; i++) {
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
    for (size_t i = 0; i < v_size; i++) {
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

float zc(std::vector<float> v, size_t v_size, float prev_sample) {
    // pred_realtime.py counts the crossing into the window's first sample too
    int zc = 0;
    float prev = prev_sample;
    for (size_t i = 0; i < v_size; i++) {
        if (prev * v[i] < 0) {
            zc++;
        }
        prev = v[i];
    }
    return zc;
}

std::vector<float> power_spectrum(const std::vector<float>& v, size_t v_size,
    const std::vector<float>& cos_table, const std::vector<float>& sin_table) {
    // direct DFT, |np.fft.rfft(v)|^2. 400 isn't a power of 2 and zero padding to 512
    // would shift the bins away from what the model was trained on
    std::vector<float> power(v_size / 2 + 1);
    for (size_t k = 0; k < power.size(); k++) {
        float re = 0.0f;
        float im = 0.0f;
        size_t idx = 0; // (k * n) mod v_size
        for (size_t n = 0; n < v_size; n++) {
            re += v[n] * cos_table[idx];
            im -= v[n] * sin_table[idx];
            idx += k;
            if (idx >= v_size) {
                idx -= v_size;
            }
        }
        power[k] = re * re + im * im;
    }
    return power;
}

float mean_freq(std::vector<float> freqs, std::vector<float> power, size_t num_bins) {
    float mean_freq = 0.0;
    for (size_t i = 0; i < num_bins; i++) {
        mean_freq += freqs[i] * power[i];
    }
    float total = std::accumulate(power.begin(), power.end(), 0.0);
    if (total <= 0.0f) {
        return 0.0f;
    }
    return mean_freq / total;
}

float median_freq(std::vector<float> freqs, std::vector<float> power, size_t num_bins) {
    // freqs[np.searchsorted(cum_power, cum_power[-1] / 2)]: first bin reaching half the power
    std::vector<float> cum_sum(num_bins);
    for (size_t i = 0; i < num_bins; i++) {
        cum_sum[i] = power[i];
        if (i != 0) {
            cum_sum[i] += cum_sum[i - 1];
        }
    }
    float half_power = cum_sum[num_bins - 1] / 2;
    for (size_t i = 0; i < num_bins; i++) {
        if (cum_sum[i] >= half_power) {
            return freqs[i];
        }
    }
    return freqs[num_bins - 1];
}

float ssc(std::vector<float> v, size_t v_size) {
    std::vector<float> diffs(v_size - 1);
    float ssc = 0.0;
    for (size_t i = 0; i < v_size - 1; i++) {
        diffs[i] = v[i+1] - v[i];
        if (i != 0) {
            if (diffs[i - 1] * diffs[i] < 0) {
                ssc++;
            }
        }
    }
    return ssc;
}

float wamp(std::vector<float> v, size_t v_size, float wamp_thresh) {
    std::vector<float> abs_diffs(v_size - 1);
    float wamp = 0.0;
    for (size_t i = 0; i < v_size - 1; i++) {
        abs_diffs[i] = std::abs(v[i+1] - v[i]);
        if (abs_diffs[i] > wamp_thresh) {
            wamp++;
        }
    }
    return wamp;
}

RealtimeFeatureExtractor::RealtimeFeatureExtractor(size_t window_sz, size_t overlap, float ref_v,
    float num_bits, float wamp_threshold) {
    window_size = window_sz;
    step_size = window_size - overlap;
    history_len = window_size + 1;
    vref = ref_v;
    bits = num_bits;
    thresh = wamp_threshold;
    // doubled so any window is one contiguous read: buffer[write_head .. write_head + history_len)
    buffer.resize(history_len * 2, 0.0f);
    features.resize(NUM_FEATURES, 0.0f);

    freqs.resize(window_size / 2 + 1, 0.0f);
    for (size_t i = 0; i < freqs.size(); i++) {
        freqs[i] = i * SAMPLE_RATE / window_size; // np.fft.rfftfreq
    }
    cos_table.resize(window_size);
    sin_table.resize(window_size);
    for (size_t i = 0; i < window_size; i++) {
        double angle = 2.0 * M_PI * i / window_size;
        cos_table[i] = std::cos(angle);
        sin_table[i] = std::sin(angle);
    }

    lowpass.configure(ButterworthFilter::LOWPASS, LOW_PASS_FREQ, SAMPLE_RATE);
    highpass.configure(ButterworthFilter::HIGHPASS, HIGH_PASS_FREQ, SAMPLE_RATE);
}

bool RealtimeFeatureExtractor::push_sample(float adc) {
    float sample = highpass.process(lowpass.process(adc_to_volts(adc, vref, bits)));

    buffer[write_head] = sample;
    buffer[history_len + write_head] = sample;

    write_head = (write_head + 1) % history_len;
    total_samples_seen++;

    // pandas rolling(window, step) evaluates at rows 0, step, 2*step..., so the first full
    // window ends at row `window` (samples 1..400), not row window - 1. Match that alignment
    if (total_samples_seen >= history_len && (total_samples_seen - history_len) % step_size == 0) {
        extract_window();
        return true;
    }
    return false;
}

void RealtimeFeatureExtractor::reset_windows() {
    total_samples_seen = 0;
    has_new_features = false;
}

const std::vector<float>& RealtimeFeatureExtractor::get_features() const { return features; }
bool RealtimeFeatureExtractor::features_ready() const {return has_new_features; }
void RealtimeFeatureExtractor::clear_features_ready() {
    has_new_features = false;
}

void RealtimeFeatureExtractor::extract_window() {
    // oldest entry is the sample just before the window (always present given the
    // alignment above)
    float prev_sample = buffer[write_head];
    std::vector<float> current_window(window_size);

    for (size_t i = 0; i < window_size; i++) {
        current_window[i] = buffer[write_head + 1 + i];
    }

    process_features(current_window, prev_sample);

}

void RealtimeFeatureExtractor::process_features(const std::vector<float>& window, float prev_sample) {
    // window is already in volts and filtered (push_sample)
    float window_rms = rms(window, window_size);
    float window_wfl = wfl(window, window_size);
    float window_std = std_dev(window, window_size);
    float window_mav = mean_abs_val(window, window_size);
    float window_min_av = min_abs_val(window, window_size);
    float window_max_av = max_abs_val(window, window_size);
    float window_zc = zc(window, window_size, prev_sample);
    float window_ssc = ssc(window, window_size);

    std::vector<float> powers = power_spectrum(window, window_size, cos_table, sin_table);
    float window_mean_freq = mean_freq(freqs, powers, freqs.size());
    float window_median_freq = median_freq(freqs, powers, freqs.size());
    // must match FEATURE_COLS order in python/train_models_v4.py (see model_params.h)
    features = {window_rms, window_wfl, window_mav, window_max_av, window_min_av,
    window_std, window_zc, window_mean_freq, window_median_freq, window_ssc};
    has_new_features = true;
}
