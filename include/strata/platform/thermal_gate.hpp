#pragma once
// Opt-in prompt-path thermal gate.  STRATA_PREFILL_TEMP_PAUSE_C=<C> makes a prompt stage wait before its next
// chunk while its GPU reads C or more, until it reads STRATA_PREFILL_TEMP_RESUME_C (default C - 3) or the cap
// STRATA_PREFILL_TEMP_MAX_WAIT_S (default 60) per chunk passes.  It only schedules work: power limits, clocks
// and fans stay untouched, and an unreadable temperature never waits.  Unset (the default): no wait, no NVML.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::platform {

struct ThermalGate {
    int pause_c = 0;            ///< 0: disabled
    int resume_c = 0;
    int step_ms = 200;          ///< sampling interval while waiting
    int max_steps = 300;        ///< per-chunk cap, from STRATA_PREFILL_TEMP_MAX_WAIT_S
};

struct ThermalWait {
    int steps = 0;              ///< sleeps taken
    int peak_c = -1, last_c = -1;
    bool capped = false, stopped = false;
};

/// The gate from `get(name)` (std::getenv by default); malformed or inconsistent settings disable it.
template <class Get>
ThermalGate thermal_gate_from(Get get) {
    auto number = [&](const char* name, double fallback, bool& ok) {
        const char* v = get(name);
        if (v == nullptr || *v == '\0') return fallback;
        char* end = nullptr;
        const double x = std::strtod(v, &end);
        if (end == v || *end != '\0' || !std::isfinite(x)) ok = false;
        return x;
    };
    bool ok = true;
    ThermalGate g;
    const double pause = number("STRATA_PREFILL_TEMP_PAUSE_C", 0, ok);
    if (pause == 0 && ok) return g;
    const double resume = number("STRATA_PREFILL_TEMP_RESUME_C", pause - 3, ok);
    const double wait_s = number("STRATA_PREFILL_TEMP_MAX_WAIT_S", 60, ok);
    if (!ok || pause < 30 || pause > 100 || resume >= pause || resume < 20 || wait_s <= 0 || wait_s > 3600 ||
        pause != std::floor(pause) || resume != std::floor(resume)) {
        std::fprintf(stderr, "strata: prompt thermal gate disabled: invalid STRATA_PREFILL_TEMP_* settings\n");
        return g;
    }
    g.pause_c = (int) pause;
    g.resume_c = (int) resume;
    g.max_steps = (int) std::ceil(wait_s * 1000.0 / g.step_ms);
    return g;
}

inline const ThermalGate& thermal_gate() {
    static const ThermalGate g = thermal_gate_from([](const char* n) { return std::getenv(n); });
    return g;
}

/// Before a prompt chunk: `read()` is the GPU temperature in C (negative: unknown), `stop()` the request's
/// cancellation, `sleep(ms)` the wait.  Waits only from `pause_c`, down to below `resume_c` or the cap.
template <class Read, class Stop, class Sleep>
ThermalWait thermal_wait(const ThermalGate& g, Read read, Stop stop, Sleep sleep) {
    ThermalWait w;
    if (g.pause_c <= 0) return w;
    int t = read();
    w.peak_c = w.last_c = t;
    if (t < g.pause_c) return w;
    while (t >= g.resume_c) {
        if (stop()) { w.stopped = true; break; }
        if (w.steps >= g.max_steps) { w.capped = true; break; }
        sleep(g.step_ms);
        ++w.steps;
        t = read();
        if (t < 0) break;
        w.last_c = t;
        if (t > w.peak_c) w.peak_c = t;
    }
    return w;
}

}  // namespace strata::platform
