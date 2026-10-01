#pragma once

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace strata::platform {

inline bool is_wsl_release(std::string release) {
    std::transform(release.begin(), release.end(), release.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return release.find("microsoft") != std::string::npos;
}

// #253: WDDM needs the multi-context cap; native Linux can register the full arena.
inline bool arena_pin_limit(bool wddm, bool multiple_devices, const char* override_gib,
                            uint64_t& bytes, std::string& err) {
    bytes = wddm && multiple_devices ? (8ull << 30) : 0;
    if (!override_gib) return true;
    const std::string_view value(override_gib);
    uint64_t gib = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), gib);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        gib > (std::numeric_limits<uint64_t>::max() >> 30)) {
        err = "STRATA_ARENA_PIN_GIB must be a non-negative integer that fits in bytes (0 = whole arena)";
        return false;
    }
    bytes = gib << 30;
    return true;
}

}  // namespace strata::platform
