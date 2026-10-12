#pragma once
// Where the engine's helper threads (every thread it creates that is neither a pool worker nor the host) run.
//   1. capture_helper_affinity() runs before the host loop pins itself: the helper mask is the process's allowed
//      CPUs minus the planned host core and its SMT siblings.  A helper never keeps the inherited single-CPU mask.
//   2. apply_helper_affinity() is the first thing a helper does: restore that mask, then let --aux-cpus
//      (strata/platform/aux_cpus.hpp) move the thread to its spare CPUs while the feature is on.  With it off the
//      second step does nothing.
//   3. restore_helper_mask() alone is for helpers whose placement is decided elsewhere (an explicit CPU pin, a thread
//      that --aux-cpus must leave alone).
// Worker pools keep their own policy.
#include "strata/platform/aux_cpus.hpp"
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
inline void restore_helper_mask() {
    if (helper_cpus_ready && sched_setaffinity(0, sizeof helper_cpus, &helper_cpus) != 0)
        std::fprintf(stderr, "strata: could not restore helper affinity\n");
}
#else
inline bool capture_helper_affinity(int) { return true; }
inline void restore_helper_mask() {}
#endif
inline void apply_helper_affinity() {
    restore_helper_mask();
    aux_cpus::pin_current_thread();
}
}
