#include "strata/core/request_number.hpp"
#include <utility>
#include <cstdio>
int main() {
 using strata::core::request_number_valid;
 int bad = 0;
 for (auto pair : {std::pair{"seed", "-1"}, {"seed", "18446744073709551616"}, {"top_k", "0"},
                   {"top_k", "65"}, {"top_k", "1.2"}, {"temperature", "nan"}, {"top_p", "inf"}, {"min_p", "1x"}})
     bad += request_number_valid(pair.first, pair.second);
 for (auto pair : {std::pair{"seed", "0"}, {"seed", "18446744073709551615"}, {"top_k", "64"}, {"temperature", "0"}})
     bad += !request_number_valid(pair.first, pair.second);
 std::printf("request predicates: %d failures\n", bad);
 return bad ? 1 : 0;
}
