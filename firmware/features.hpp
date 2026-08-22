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

float sum_first_nvoid(int *arr, int num_sum, float sum) {
    int n = 0;
    if (num_sum <= n) {
        return sum;
    }
    else {
        sum += arr[n];
        n++;
        sum_first_nvoid(arr, num_sum, sum);
    }
}


class RealtimeFeatureExtractor {
private:
    size_t window_size;
    std::vector<double> buffer;
    size_t head = 0;
    size_t count = 0;
    double sum = 0.0;
    double sq_sum = 0.0; // Sum of squares for variance

public:
    explicit RealtimeFeatureExtractor(size_t k, size_t step) : window_size(k), buffer(k, 0.0) {}

    // Call this function for every new incoming real-time sample
    void update(double newValue) {
        if (count < window_size) {
            buffer[head] = newValue;
            sum += newValue;
            sq_sum += newValue * newValue;
            count++;
        } else {
            // Remove oldest step values from running totals
            double oldValue = buffer[head];
            sum -= oldValue;
            sq_sum -= oldValue * oldValue;

            // Insert new value
            buffer[head] = newValue;
            sum += newValue;
            sq_sum += newValue * newValue;
        }

        // Advance circular buffer head index
        head = (head + 1) % window_size;
    }

    double getMean() const {
        if (count == 0) return 0.0;
        return sum / count;
    }

    double getVariance() const {
        if (count <= 1) return 0.0;
        double mean = getMean();
        // Variance formula: E[X^2] - (E[X])^2
        double var = (sq_sum / count) - (mean * mean);
        return var < 0.0 ? 0.0 : var; // Avoid negative float errors
    }

    double getStandardDeviation() const {
        return std::sqrt(getVariance());
    }
    
    bool isReady() const {
        return count == window_size;
    }
};