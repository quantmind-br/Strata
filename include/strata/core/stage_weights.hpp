#pragma once
#include <string>
#include <cstdlib>
namespace strata::core {
inline bool stage_weight_needed(const std::string& name, int64_t begin, int64_t end) {
    if (end < 0 || name.rfind("blk.", 0) != 0 || name == "blk.1.ple_key.weight") return true;
    char* tail = nullptr;
    const auto layer = std::strtoll(name.c_str() + 4, &tail, 10);
    return !tail || *tail != '.' || (layer >= begin && layer < end);
}
}
