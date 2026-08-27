#include <stdio.h>
#include <string.h>
#include <vector>
#include <numeric>
#include <cmath>

const int LOW_PASS_FREQ = 500;
const int HIGH_PASS_FREQ = 20;
const int SAMPLE_RATE = 2000; // Hz
const int WINDOW = 400;
const int STEP = 200;

const int BITS = 10;
const int VREF = 3.7;


void remove_first_n(int *arr, int *current_size, int n) {
    if (n >= *current_size) {
        *current_size = 0; // All elements removed
        return;
    }

    // Number of elements that will remain
    int remaining = *current_size - n;

    // Shift remaining elements to the beginning of the array
    memmove(arr, arr + n, remaining * sizeof(int));

    // Update the logical size tracker
    *current_size = remaining;
}

float sum_first_n(int *arr, int num_sum, float sum) {
    int n = 0;
    if (num_sum <= n) {
        return sum;
    }
    else {
        sum += arr[n];
        n++;
        sum_first_n(arr, num_sum, sum);
    }
}



class RealtimeFeatureExtractor {
private:
    size_t window_size;
    size_t step;
    std::vector<float> buffer;
    size_t write_index = 0;
    size_t filled_samples = 0;

public:
    RealtimeFeatureExtractor(size_t window_size, size_t step) 
        : window_size(window_size), 
          step(step), 
          buffer(window_size, 0.0f) {}

    // feed incoming data chunk by chunk,
    // return true if we've fillwed the window
    bool update(float incoming_sample, std::vector<float>& out_frame) {
        buffer[write_index] = incoming_sample;
        write_index = (write_index + 1) % window_size;
        
        if (filled_samples < window_size) {
            filled_samples++;
            return false; // first window not yet filled
        }
        return true; 
    }

    // Block-based real-time update (more efficient)
    bool push_block(const std::vector<float>& input_block, std::vector<std::vector<float>>& extracted_frames) {
        for (float sample : input_block) {
            buffer[write_index] = sample;
            write_index = (write_index + 1) % window_size;
            
            if (filled_samples < window_size) {
                filled_samples++;
            } else {
                // When buffer wraps or hits hop cadence, reorder and push frame
                // (In practice, use a FIFO history buffer of size frame_size + block_size)
            }
        }
        return false;
    }
};