#pragma once
// Capture before the host loop pins itself. Helpers restore the caller's allowed
// mask, never the inherited single-CPU mask. Worker pools keep their own policy.
#if !defined(_WIN32)
#include <sched.h>
#include <cstdio>
#endif
namespace strata::platform {
#if !defined(_WIN32)
inline cpu_set_t helper_cpus{};
inline bool helper_cpus_ready = false;
inline bool capture_helper_affinity(int host_cpu) {
    if (helper_cpus_ready) return true;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return false;
    auto topology = [](int cpu, const char* item) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, item);
        int value = -1;
        if (auto* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%d", &value) != 1) value = -1;
            std::fclose(f);
        }
        return value;
    };
    const int core = topology(host_cpu, "core_id"), package = topology(host_cpu, "physical_package_id");
    auto helpers = allowed;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &helpers) && (cpu == host_cpu || (core >= 0 && package >= 0 &&
            topology(cpu, "core_id") == core && topology(cpu, "physical_package_id") == package)))
            CPU_CLR(cpu, &helpers);
    helper_cpus = CPU_COUNT(&helpers) ? helpers : allowed;
    helper_cpus_ready = true;
    return true;
}
inline void apply_helper_affinity() {
    if (helper_cpus_ready && sched_setaffinity(0, sizeof helper_cpus, &helper_cpus) != 0)
        std::fprintf(stderr, "strata: could not restore helper affinity\n");
}
#else
inline bool capture_helper_affinity(int) { return true; }
inline void apply_helper_affinity() {}
#endif
}
