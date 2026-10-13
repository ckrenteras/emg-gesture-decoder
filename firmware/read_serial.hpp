#pragma once
#include <vector>
#include "feature_extraction.hpp"

extern RealtimeFeatureExtractor extractor;

void init_serial();
void start_sampling();  // starts the 2 kHz ADC timer on the EMG pin
void flush_samples();   // drops any samples buffered but not yet read, restarts windowing
void read_available_samples();
