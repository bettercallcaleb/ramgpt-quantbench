#pragma once
#include <cstdint>
#include <cstddef>

namespace rq {
struct RowMetrics {
    int32_t ref_top1, ref_top2, quant_top1, quant_top2;
    double ref_top1_logit, ref_top2_logit, quant_top1_logit, quant_top2_logit;
    double ref_margin, kl, p_ref_target, p_quant_target, delta_p_target;
    double nll_ref, nll_quant, delta_nll;
    bool decision_flip;
};
RowMetrics calculate_metrics(const float * ref, const float * quant, uint32_t vocab, int32_t target);
double percentile_sorted(const double * values, size_t count, double percentile);
}
