#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::prefill {

enum class InitResult { ready, out_of_memory, error };

// A failed stage invalidates the whole pipeline. Release even the failing stage's partial allocations
// before retrying every stage at one shared chunk size. Callbacks select each stage's device.
template <class Init, class Release, class Retry>
bool init_pipeline(int64_t& chunk, bool automatic, size_t stages, Init init, Release release, Retry retry) {
    for (;;) {
        InitResult result = InitResult::ready;
        size_t stage = 0;
        for (; stage < stages; ++stage) {
            result = init(stage, chunk);
            if (result != InitResult::ready) break;
        }
        if (result == InitResult::ready) return true;
        for (size_t i = stages; i > 0; --i) release(i - 1);
        if (!automatic || result != InitResult::out_of_memory || chunk <= 256) return false;
        const int64_t smaller = std::max<int64_t>(256, chunk / 2);
        retry(stage, chunk, smaller);
        chunk = smaller;
    }
}

}  // namespace strata::prefill
