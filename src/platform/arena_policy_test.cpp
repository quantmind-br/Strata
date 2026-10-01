#include "strata/platform/arena_policy.hpp"

#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace strata::platform;
    uint64_t bytes = 0;
    std::string err;
    for (bool multi : {false, true}) {
        require(arena_pin_limit(false, multi, nullptr, bytes, err) && bytes == 0,
                "native Linux must attempt full registration");
        require(arena_pin_limit(true, multi, nullptr, bytes, err) && bytes == (multi ? 8ull << 30 : 0),
                "WDDM cap must apply only with multiple devices");
        for (bool wddm : {false, true}) {
            require(arena_pin_limit(wddm, multi, "0", bytes, err) && bytes == 0, "explicit full registration");
            require(arena_pin_limit(wddm, multi, "12", bytes, err) && bytes == 12ull << 30, "explicit cap");
        }
    }
    for (const char* value : {"", "-1", "+1", " 8", "8 ", "1.5", "8GiB", "17179869184", "99999999999999999999999"}) {
        err.clear();
        require(!arena_pin_limit(false, true, value, bytes, err) && !err.empty(), "invalid override accepted");
    }
    require(arena_pin_limit(false, true, "17179869183", bytes, err) && bytes == 17179869183ull << 30,
            "largest representable cap rejected");
    require(is_wsl_release("6.6.87.2-microsoft-standard-WSL2"), "WSL2 not detected");
    require(is_wsl_release("4.4.0-Microsoft"), "mixed-case WSL not detected");
    require(!is_wsl_release("6.18.3-arch1-1"), "native Linux detected as WSL");
    std::cout << "arena_policy_test: OK\n";
}
