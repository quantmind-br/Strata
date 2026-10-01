#include "strata/prefill/startup.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace strata::prefill;
    // Either GPU can exhaust memory after partially allocating its buffers. The next attempt must start clean.
    for (size_t failing_stage : {0u, 1u}) {
        int64_t chunk = 2048;
        std::vector<int64_t> allocated(2);
        std::vector<size_t> released;
        int retries = 0;
        require(init_pipeline(chunk, true, 2,
            [&](size_t stage, int64_t size) {
                if (stage == 0) require(allocated == std::vector<int64_t>({0, 0}), "retry leaked an allocation");
                allocated[stage] = size;  // includes partial allocation on the failing device
                return stage == failing_stage && size > 512 ? InitResult::out_of_memory : InitResult::ready;
            },
            [&](size_t stage) { released.push_back(stage); allocated[stage] = 0; },
            [&](size_t stage, int64_t from, int64_t to) {
                require(stage == failing_stage && to == from / 2, "wrong retry decision");
                require(allocated == std::vector<int64_t>({0, 0}), "retry preceded cleanup");
                ++retries;
            }), "automatic retry failed");
        require(chunk == 512 && retries == 2, "wrong final chunk");
        require(allocated == std::vector<int64_t>({512, 512}), "pipeline has mixed chunk sizes");
        require(released == std::vector<size_t>({1, 0, 1, 0}), "incomplete cleanup of all stages");
    }
    for (bool automatic : {false, true}) {
        for (InitResult failure : {InitResult::error, InitResult::out_of_memory}) {
            int64_t chunk = 1024;
            int attempts = 0, releases = 0, retries = 0;
            require(!init_pipeline(chunk, automatic, 2,
                [&](size_t, int64_t) { ++attempts; return failure; },
                [&](size_t) { ++releases; },
                [&](size_t, int64_t, int64_t) { ++retries; }), "persistent failure accepted");
            const bool reduce = automatic && failure == InitResult::out_of_memory;
            require(chunk == (reduce ? 256 : 1024), "explicit chunk or non-memory error retried");
            require(attempts == (reduce ? 3 : 1) && releases == 2 * attempts && retries == attempts - 1,
                    "wrong stop/cleanup behavior");
        }
    }
    int64_t chunk = 300;
    std::vector<int64_t> tried;
    require(!init_pipeline(chunk, true, 1,
        [&](size_t, int64_t size) { tried.push_back(size); return InitResult::out_of_memory; },
        [](size_t) {}, [](size_t, int64_t, int64_t) {}), "minimum failure accepted");
    require(tried == std::vector<int64_t>({300, 256}), "minimum chunk skipped");
    std::cout << "prefill_startup_test: OK\n";
}
