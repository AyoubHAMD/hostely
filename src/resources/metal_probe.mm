// hostely Metal memory probe — single Obj-C++ translation unit.
//
// We need Objective-C++ here because Metal's API is not exposed through a C
// interface: `MTLCreateSystemDefaultDevice()` returns `id<MTLDevice>`, and
// the device's `-currentAllocatedSize` / `-recommendedMaxWorkingSetSize` are
// Objective-C selectors.
//
// The rest of the codebase calls into this file through the plain C ABI
// declared in metal_probe.hpp, so no other translation unit needs to be Obj-C++.

#include "resources/metal_probe.hpp"

#import <Metal/Metal.h>

#include <cstring>

namespace {

// Lazily resolve and cache the default Metal device for the process.
// On Apple Silicon this is the integrated GPU; on Intel Macs with no
// Metal device, this returns nil.
id<MTLDevice> get_device() {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
    });
    return dev;
}

}  // namespace

extern "C" int hostely_metal_available(void) {
    return get_device() != nil ? 1 : 0;
}

extern "C" unsigned long long hostely_metal_current_allocated_bytes(void) {
    id<MTLDevice> dev = get_device();
    if (dev == nil) return 0ULL;
    @try {
        return static_cast<unsigned long long>([dev currentAllocatedSize]);
    } @catch (...) {
        return 0ULL;
    }
}

extern "C" unsigned long long hostely_metal_recommended_max_bytes(void) {
    id<MTLDevice> dev = get_device();
    if (dev == nil) return 0ULL;
    @try {
        return static_cast<unsigned long long>([dev recommendedMaxWorkingSetSize]);
    } @catch (...) {
        return 0ULL;
    }
}

extern "C" const char* hostely_metal_device_name(char* out, int out_len) {
    if (out == nullptr || out_len <= 0) return "";
    out[0] = '\0';
    id<MTLDevice> dev = get_device();
    if (dev == nil) return out;
    NSString* name = [dev name];
    if (name == nil) return out;
    const char* utf8 = [name UTF8String];
    if (utf8 == nullptr) return out;
    std::strncpy(out, utf8, static_cast<size_t>(out_len) - 1);
    out[out_len - 1] = '\0';
    return out;
}
