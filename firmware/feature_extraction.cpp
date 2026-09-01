#include "feature_extraction.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <arm_math.h> 


 std::vector<float> adc_to_volts( std::vector<float> adc, float vref, float bits) {
    for (float &elt : adc) {
        elt = ((elt * vref) / (std::pow(2, bits) - 1));
    }
    return adc;
}

float rms(std::vector<float> v, size_t v_size) {
    float mean_sqr = 0.0;
    for (int i = 0; i < v_size; i++) {
        mean_sqr += std::pow(v[i], 2);
    } 
    mean_sqr /= v_size;
    return std::pow(mean_sqr, 0.5);
}

float wfl(std::vector<float> v, size_t v_size) {
    float len = 0.0;
    for (int i = 0; i < v_size - 1; i++) {
        len += std::abs(v[i + 1] - v[i]);
    }
    return len;
}

float std_dev(std::vector<float> v, size_t v_size) {
    float mean = std::accumulate(v.begin(), v.end(), 0.0) / v_size;
    float var = 0.0;
    for (int i = 0;  i < v_size; i++) {
        var += std::pow(v[i] - mean, 2);
    }
    var /= v_size;
    return std::pow(var, 0.5);
}

float mean_abs_val(std::vector<float> v, size_t v_size) {
   float mean_abs = 0.0;
    for (int i = 0; i < v_size; i++) {
        mean_abs += v[i];
    }
    return mean_abs /= v_size;
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

float zc(std::vector<float> v, size_t v_size) {
    int zc = 0;
    float prev;
    for (int i = 0; i < v_size; i++) {
        if (i !=0) {
            if (prev * v[i] < 0) {
                zc++;
            } 
        }
        prev = v[i];
    }
    return zc;
}

std::vector<float> fourier_transf(std::vector<float> input_v, size_t v_size, arm_cfft_instance_f32& fft_instance) {
    /// input_v should be a copy of the window, it will get overwritten
    // magnitudes will be saved to target_v, freqs in place to inpiut_v
    std::vector<float> buffer_v(FFT_SIZE * 2);
    for (int i = 0; i < v_size; i++) {
        buffer_v[2*i] = input_v.data()[i];
    }
    arm_cfft_f32(&fft_instance, buffer_v.data(), 0, 1);
    std::vector<float> magnitudes(FFT_SIZE);
    arm_cmplx_mag_f32(buffer_v.data(), magnitudes.data(), FFT_SIZE);
    return magnitudes;
}

 std::vector<float> power_vector(std::vector<float> magnitudes, size_t half_fft_sz) {
    // computes a vector whose entries sum to power
     std::vector<float> target(half_fft_sz);
    for (int i = 0; i < half_fft_sz; i++) {
        target[i] = std::pow(std::abs(magnitudes[i]), 2);
    }
    return target;
}

float mean_freq(std::vector<float> freqs, std::vector<float> power, size_t half_fft_sz) {
    float mean_freq = 0.0;
    for (int i = 0; i < half_fft_sz; i++) {
        mean_freq += freqs[i] * power[i];
    }
    mean_freq /= std::accumulate(power.begin(), power.end(), 0.0);
    return mean_freq;
}

float median_freq(std::vector<float> freqs, std::vector<float> power, size_t half_fft_sz) {
    std::vector<float> cum_sum(half_fft_sz);
    for (int i = 0; i < half_fft_sz; i++) {
        cum_sum[i] = power[i];
        if (i != 0) {
            cum_sum[i] += cum_sum[i - 1];
        }
    }
    float half_power = cum_sum[half_fft_sz - 1] / 2;
    float median_freq;
    for (int i = 0; i < half_fft_sz; i++) {
        if (cum_sum[i] < half_power && cum_sum[i+1] >= half_power) {
            median_freq = freqs[i+1];
            break;
        }
    }
    return median_freq;
}

float ssc(std::vector<float> v, size_t v_size) {
    std::vector<float> diffs(v_size - 1);
    float ssc = 0.0;
    for (int i = 0; i < v_size - 1; i++) {
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
    for (int i = 0; i < v_size - 1; i++) {
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
    vref = ref_v;
    bits = num_bits;
    thresh = wamp_threshold;
    buffer.resize(window_size * 2, 0.0f);
    features.resize(11, 0.0f);
    arm_cfft_init_f32(&fft_instance, FFT_SIZE, 0, 1);
    freqs.resize(FFT_SIZE / 2, 0.0f);
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        freqs[i] = i * SAMPLE_RATE / FFT_SIZE;
    }
}

void RealtimeFeatureExtractor::read_samples(const std::vector<float>& data) {
    if (has_new_features) {
        clear_has_new_features();
    }
    for (float sample : data) {
        buffer[write_head] = sample;
        buffer[window_size + write_head] = sample;

        write_head = (write_head + 1) % window_size;
        num_samples_read++;
        total_samples_seen++;

        if (total_samples_seen >= window_size && !buffer_filled) {
            buffer_filled = true;
        }

       if (num_samples_read >= step_size && buffer_filled) {
            extract_window();
            num_samples_read = 0;
        }
    }
}

const std::vector<float>& RealtimeFeatureExtractor::get_features() const { return features; }
bool RealtimeFeatureExtractor::features_ready() const {return has_new_features; }
void RealtimeFeatureExtractor::clear_features_ready() { 
    has_new_features = false;
}

void RealtimeFeatureExtractor::extract_window() {
    std::vector<float> current_window(window_size);

    for (size_t i = 0; i < window_size; i++) {
        current_window[i] = buffer[write_head + i];
    }

    process_features(current_window);

}

void RealtimeFeatureExtractor::process_features(const std::vector<float>& window) {
    // insert features here
    std::vector<float> emg = adc_to_volts(window, vref, bits);
    std::vector<float> emg_cpy = emg;
    float window_rms = rms(emg_cpy, window_size);
    float window_wfl = wfl(emg_cpy, window_size);
    float window_std = std_dev(emg_cpy, window_size);
    float window_mav = mean_abs_val(emg_cpy, window_size);
    float window_min_av = min_abs_val(emg_cpy, window_size);
    float window_max_av = max_abs_val(emg_cpy, window_size);
    float window_zc = zc(emg_cpy, window_size);
    float window_ssc = ssc(emg_cpy, window_size);
    float window_wamp = wamp(emg_cpy, window_size, thresh);

    std::vector<float> magnitudes = fourier_transf(emg, window_size, fft_instance);
    std::vector<float> powers = power_vector(magnitudes, FFT_SIZE / 2);
    float window_mean_freq = mean_freq(freqs, powers, FFT_SIZE / 2);
    float window_median_freq = median_freq(freqs, powers, FFT_SIZE / 2);
    features = {window_rms, window_wfl, window_std, window_mav, window_min_av, 
    window_max_av, window_zc, window_ssc, window_wamp, window_mean_freq, window_median_freq};
    has_new_features = true;
}