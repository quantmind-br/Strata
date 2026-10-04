#pragma once
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
namespace strata::core {
inline bool request_number_valid(const std::string& key, const std::string& value) {
    if (value.empty()) return false;
    if (key == "seed") {
        if (value.find_first_not_of("0123456789") != std::string::npos) return false;
        errno = 0;
        char* end = nullptr;
        std::strtoull(value.c_str(), &end, 10);
        return errno != ERANGE && end && *end == '\0';
    }
    errno = 0;
    char* end = nullptr;
    double n = std::strtod(value.c_str(), &end);
    if (errno == ERANGE || !end || *end != '\0' || !std::isfinite(n)) return false;
    if (key == "top_k") return value.find_first_not_of("0123456789") == std::string::npos && n >= 1 && n <= 64;
    if (key == "penalty_last_n") return value.find_first_not_of("0123456789") == std::string::npos && n >= 0 && n <= 2147483647;
    if (key == "cvec") return value == "0" || value == "1";
    if (key == "temperature" || key == "penalty_freq" || key == "penalty_present") return n >= 0 && n <= 2;
    if (key == "top_p") return n > 0 && n <= 1;
    if (key == "min_p" || key == "pcie_frac" || key == "spec_min_p") return n >= 0 && n <= 1;
    if (key == "penalty_repeat") return n > 0 && n <= 1e6;
    return false;
}
}
