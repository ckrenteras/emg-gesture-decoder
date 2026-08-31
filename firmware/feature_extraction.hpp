#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <arm_math.h> 

const float LOW_PASS_FREQ = 500;
const float HIGH_PASS_FREQ = 20;
const float SAMPLE_RATE = 2000; // Hz
const size_t WINDOW = 400;
const size_t STEP = 200;
const size_t FFT_SIZE = 512; // 400 samples, minimum pow of 2 to fit that many is 512
const float WAMP_THRESH = 0.003;
const int NUM_FEATURES = 11;

const float BITS = 10;
const float VREF = 3.7;


// features
std::vector<float> adc_to_volts( std::vector<float> adc, float vref, float bits);

float rms(std::vector<float> v, size_t v_size);
float wfl(std::vector<float> v, size_t v_size);
float std_dev(std::vector<float> v, size_t v_size);
float mean_abs_val(std::vector<float> v, size_t v_size);
float min_abs_val(std::vector<float> v, size_t v_size);
float max_abs_val(std::vector<float> v, size_t v_size);
float zc(std::vector<float> v, size_t v_size);

std::vector<float> fourier_transf(std::vector<float> input_v, size_t v_size, arm_cfft_instance_f32& fft_instance);

 std::vector<float> power_vector(std::vector<float> magnitudes, size_t v_size);

float mean_freq(std::vector<float> freqs, std::vector<float> power, size_t v_size);

float median_freq(std::vector<float> freqs, std::vector<float> power, size_t v_size);

float ssc(std::vector<float> v, size_t v_size);

float wamp(std::vector<float> v, size_t v_size, float wamp_thresh);

class RealtimeFeatureExtractor {
private:
    size_t window_size;
    size_t step_size;
    std::vector<float> buffer;
    size_t write_head = 0;
    size_t num_samples_read = 0;
    float vref;
    float bits;
    float thresh;
    std::vector<float> features;
    size_t total_samples_seen = 0;
    bool buffer_filled = false;
    std::vector<float> freqs;
    arm_cfft_instance_f32 fft_instance;
public:
    RealtimeFeatureExtractor(size_t window_sz, size_t overlap, float ref_v, float num_bits, float wamp_threshold);

    void read_samples(const std::vector<float>& data) ;
private:
    void extract_window();
    void process_features(const std::vector<float>& window);
};