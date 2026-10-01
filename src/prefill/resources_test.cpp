// Model-free CUDA lifecycle checks. Link wrapping injects failures and counts direct owned allocations.
#include "strata/prefill/prefill.hpp"
#include "strata/core/pinned.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

static std::unordered_set<void*> device_buffers, host_buffers, events, streams;
static int fail_malloc = 0;
static cudaError_t injected_failure = cudaErrorMemoryAllocation;
static bool refuse_full_arena = false;
static constexpr size_t arena_bytes = 2u << 20;

extern "C" {
cudaError_t __real_cudaMalloc(void**, size_t);
cudaError_t __wrap_cudaMalloc(void** ptr, size_t bytes) {
    if (fail_malloc > 0 && --fail_malloc == 0) return injected_failure;
    const auto status = __real_cudaMalloc(ptr, bytes);
    if (status == cudaSuccess) device_buffers.insert(*ptr);
    return status;
}
cudaError_t __real_cudaFree(void*);
cudaError_t __wrap_cudaFree(void* ptr) {
    const auto status = __real_cudaFree(ptr);
    if (status == cudaSuccess) device_buffers.erase(ptr);
    return status;
}
cudaError_t __real_cudaHostAlloc(void**, size_t, unsigned);
cudaError_t __wrap_cudaHostAlloc(void** ptr, size_t bytes, unsigned flags) {
    const auto status = __real_cudaHostAlloc(ptr, bytes, flags);
    if (status == cudaSuccess) host_buffers.insert(*ptr);
    return status;
}
cudaError_t __real_cudaFreeHost(void*);
cudaError_t __wrap_cudaFreeHost(void* ptr) {
    const auto status = __real_cudaFreeHost(ptr);
    if (status == cudaSuccess) host_buffers.erase(ptr);
    return status;
}
cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned);
cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    const auto status = __real_cudaEventCreateWithFlags(event, flags);
    if (status == cudaSuccess) events.insert(*event);
    return status;
}
cudaError_t __real_cudaEventDestroy(cudaEvent_t);
cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
    const auto status = __real_cudaEventDestroy(event);
    if (status == cudaSuccess) events.erase(event);
    return status;
}
cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags) {
    const auto status = __real_cudaStreamCreateWithFlags(stream, flags);
    if (status == cudaSuccess) streams.insert(*stream);
    return status;
}
cudaError_t __real_cudaStreamDestroy(cudaStream_t);
cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
    const auto status = __real_cudaStreamDestroy(stream);
    if (status == cudaSuccess) streams.erase(stream);
    return status;
}
cudaError_t __real_cudaHostRegister(void*, size_t, unsigned);
cudaError_t __wrap_cudaHostRegister(void* ptr, size_t bytes, unsigned flags) {
    if (refuse_full_arena && bytes == arena_bytes) return cudaErrorMemoryAllocation;
    return __real_cudaHostRegister(ptr, bytes, flags);
}
}

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace strata;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    core::WeightTable weights;
    core::ModelGeometry geometry;
    core::QsaState qsa{};
    qsa.max_cells = 1024;
    core::SessionState session;
    session.qsa_states = &qsa;
    for (int dev = 0; dev < devices; ++dev) {
        require(cudaSetDevice(dev) == cudaSuccess, "select test GPU");
        for (int fail_at : {1, 2, 4, -2, 0}) {
            prefill::Prefill prompt;
            std::string error;
            fail_malloc = fail_at < 0 ? -fail_at : fail_at;
            injected_failure = fail_at < 0 ? cudaErrorInvalidValue : cudaErrorMemoryAllocation;
            const bool ready = prompt.init(weights, geometry, session, nullptr, nullptr, nullptr,
                                           256, nullptr, error);
            require(ready == (fail_at == 0), error.c_str());
            const auto failure = fail_at < 0 ? prefill::InitResult::error : prefill::InitResult::out_of_memory;
            require(prompt.init_result() == (ready ? prefill::InitResult::ready : failure),
                    "allocation failure classified incorrectly");
            if (fail_at < 0) require(error.find(cudaGetErrorString(injected_failure)) != std::string::npos,
                                     "original CUDA error lost");
            // Cleanup must select the owning device and restore the caller's device.
            const int caller = (dev + 1) % devices;
            require(cudaSetDevice(caller) == cudaSuccess, "select caller GPU");
            prompt.reset();
            prompt.reset();
            int current = -1;
            require(cudaGetDevice(&current) == cudaSuccess && current == caller, "reset changed caller device");
            require(device_buffers.empty() && host_buffers.empty() && events.empty() && streams.empty(),
                    "prefill reset leaked buffers, events or streams");
            require(cudaGetLastError() == cudaSuccess, "stale CUDA error after reset");
            require(cudaSetDevice(dev) == cudaSuccess, "restore test GPU");
        }
        for (bool fallback : {false, true}) {
            refuse_full_arena = fallback;
            core::PinnedArena arena(arena_bytes, std::vector<uint64_t>{0, arena_bytes / 2, arena_bytes});
            require(arena.valid() && arena.registered_bytes == arena_bytes, "full/sliced registration failed");
            require(arena.registered_slices == (fallback ? 2 : 0), "registration fallback did not run");
            require(cudaGetLastError() == cudaSuccess, "registration left a stale CUDA error");
        }
        refuse_full_arena = false;
        core::PinnedArena capped(arena_bytes, std::vector<uint64_t>{0, arena_bytes / 2, arena_bytes}, arena_bytes / 2);
        require(capped.valid() && capped.registered_bytes == arena_bytes / 2, "registration ignored cap");
        capped.data()[arena_bytes - 1] = 42;
        require(capped.data()[arena_bytes - 1] == 42, "unregistered experts are not accessible");
    }
    std::puts("prefill_resources_test: OK");
}
