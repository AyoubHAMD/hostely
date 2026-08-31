#pragma once

#include <cstdint>

// Metal memory probe — the one place in hostely where we use Objective-C++
// because Metal's API is not exposed through a C interface.
//
// Exported as a plain C ABI so the rest of the codebase can link without
// pulling in <Metal/Metal.h>. Build the .mm file with `-x objective-c++`.

extern "C" {

/// True iff a Metal device exists on this machine (Apple Silicon Macs always
/// have one; Intel iGPUs may not).
int hostely_metal_available(void);

/// Total bytes the GPU is currently holding across all allocations made by
/// *this process*. Returns 0 if Metal is unavailable or the call fails.
unsigned long long hostely_metal_current_allocated_bytes(void);

/// Recommended maximum working-set size, in bytes. Above this, the OS may
/// evict allocations. Apple Silicon typically reports total unified memory
/// here (e.g. ~22 GB on a 32 GB M-series). Returns 0 on failure.
unsigned long long hostely_metal_recommended_max_bytes(void);

/// Human-readable GPU name ("Apple M4"). Caller owns the returned buffer;
/// it is at most 64 bytes including NUL.
const char* hostely_metal_device_name(char* out, int out_len);

}  // extern "C"
