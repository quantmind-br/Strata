#include "strata/platform/thermal_gate.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {
using strata::platform::ThermalGate;
using strata::platform::thermal_gate_from;
using strata::platform::thermal_wait;

ThermalGate gate(std::map<std::string, std::string> env) {
    return thermal_gate_from([&](const char* n) { auto it = env.find(n); return it == env.end() ? nullptr : it->second.c_str(); });
}

struct Run { int steps; int peak; int last; bool capped; bool stopped; };

Run run(const ThermalGate& g, std::vector<int> temps, int stop_after = -1) {
    size_t i = 0;
    int reads = 0;
    auto w = thermal_wait(g, [&] { ++reads; return i < temps.size() ? temps[i++] : temps.back(); },
                          [&] { return stop_after >= 0 && reads > stop_after; }, [](int) {});
    return {w.steps, w.peak_c, w.last_c, w.capped, w.stopped};
}
}  // namespace

int main() {
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::printf("FAIL %s\n", what); ++bad; } };
    expect(gate({}).pause_c == 0, "unset is disabled");
    expect(run(gate({}), {99}).steps == 0, "disabled never waits");
    const ThermalGate g = gate({{"STRATA_PREFILL_TEMP_PAUSE_C", "80"}});
    expect(g.pause_c == 80 && g.resume_c == 77 && g.max_steps == 300, "defaults");
    expect(run(g, {79}).steps == 0, "below the pause threshold");
    expect(run(g, {-1}).steps == 0, "unknown temperature");
    Run r = run(g, {82, 83, 79, 77, 76});
    expect(r.steps == 4 && r.peak == 83 && r.last == 76 && !r.capped, "waits below resume");
    r = run(g, {81, -1});
    expect(r.steps == 1 && !r.capped, "lost telemetry ends the wait");
    r = run(gate({{"STRATA_PREFILL_TEMP_PAUSE_C", "80"}, {"STRATA_PREFILL_TEMP_MAX_WAIT_S", "1"}}), {90});
    expect(r.capped && r.steps == 5, "per-chunk cap");
    r = run(g, {85}, 1);
    expect(r.stopped && r.steps == 1, "cancellation ends the wait");
    for (auto env : std::vector<std::map<std::string, std::string>>{
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "80x"}},
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "80.5"}},
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "120"}},
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "80"}, {"STRATA_PREFILL_TEMP_RESUME_C", "80"}},
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "80"}, {"STRATA_PREFILL_TEMP_MAX_WAIT_S", "0"}},
             {{"STRATA_PREFILL_TEMP_PAUSE_C", "nan"}}})
        expect(gate(env).pause_c == 0, "invalid settings disable the gate");
    std::printf("thermal gate: %d failures\n", bad);
    return bad ? 1 : 0;
}
