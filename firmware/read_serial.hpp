#pragma once
#include <vector>
#include "feature_extraction.hpp"

extern RealtimeFeatureExtractor extractor;

void init_serial();
void read_available_samples();