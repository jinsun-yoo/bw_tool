#pragma once
#include <atomic>
#include "sampler.h"

void writer_thread(SampleBuffer* buf, std::atomic<bool>* stop, const char* csv_path);
