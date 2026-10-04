#include "strata/core/request_number.hpp"
#include "strata/core/stage_weights.hpp"
#include <cstdio>
int main() {
 using strata::core::request_number_valid;
 int bad = 0;
 for (auto pair : {std::pair{"seed", "-1"}, {"seed", "18446744073709551616"}, {"top_k", "0"},
                   {"top_k", "65"}, {"top_k", "1.2"}, {"temperature", "nan"}, {"top_p", "inf"}, {"min_p", "1x"}})
     bad += request_number_valid(pair.first, pair.second);
 for (auto pair : {std::pair{"seed", "0"}, {"seed", "18446744073709551615"}, {"top_k", "64"}, {"temperature", "0"}})
     bad += !request_number_valid(pair.first, pair.second);
 using strata::core::stage_weight_needed;
 bad += stage_weight_needed("blk.24.attn_q.weight", 0, 24);
 bad += !stage_weight_needed("blk.24.attn_q.weight", 24, 48);
 bad += !stage_weight_needed("blk.1.ple_key.weight", 24, 48);
 bad += !stage_weight_needed("output.weight", 24, 48);
 std::printf("request/stage predicates: %d failures\n", bad);
 return bad ? 1 : 0;
}
