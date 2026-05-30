#include "routed_predictor.h"

#include <cstdio>
#include <cstring>

// Single definition of the global used by cond_branch_predictor_interface.cc
RoutedPredictor cond_predictor_impl;

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool RoutedPredictor::gshare_pred_now(uint64_t pc) const {
    int idx = (int)(((pc >> 2) ^ fast_ghr_) & ((1u << SLOG) - 1));
    return gshare_[idx] >= 4;
}

bool RoutedPredictor::router_invoke(uint64_t pc) const {
    uint32_t mask = (1u << RLOG) - 1;
    uint32_t ghr  = fast_ghr_;
    int sum = router_[0][(pc >> 2) & mask]                            // bias
            + router_[1][((pc >> 2) ^ ghr) & mask]                    // PC ^ GHR[9:0]
            + router_[2][((pc >> 2) ^ (ghr >> 5)) & mask]             // PC ^ GHR[14:5]
            + router_[3][((pc >> 2) ^ (ghr ^ (ghr >> RLOG))) & mask]; // PC ^ fold(GHR)
    return sum > RTHETA;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

RoutedPredictor::RoutedPredictor()
    : fast_ghr_(0),
      stat_total_(0), stat_fast_used_(0),
      stat_fast_correct_(0), stat_tage_correct_(0),
      stat_disagree_(0),
      stat_disagree_tage_right_(0), stat_disagree_fast_right_(0)
{
    memset(bimodal_, 2, sizeof(bimodal_));  // weakly taken
    memset(gshare_,  4, sizeof(gshare_));   // weakly taken
    memset(router_,  1, sizeof(router_));   // sum = RTABS*1 > RTHETA → invoke TAGE by default
}

void RoutedPredictor::setup() {}

void RoutedPredictor::terminate()
{
    if (stat_total_ == 0) return;

    uint64_t tage_used = stat_total_ - stat_fast_used_;
    double pct_fast      = 100.0 * stat_fast_used_ / stat_total_;
    double pct_tage      = 100.0 * tage_used        / stat_total_;
    double pct_disagr    = 100.0 * stat_disagree_   / stat_total_;
    double fast_acc      = stat_fast_used_ > 0
                         ? 100.0 * stat_fast_correct_ / stat_fast_used_ : 0.0;
    double tage_acc      = tage_used > 0
                         ? 100.0 * stat_tage_correct_ / tage_used : 0.0;
    double dis_tage_pct  = stat_disagree_ > 0
                         ? 100.0 * stat_disagree_tage_right_ / stat_disagree_ : 0.0;
    double dis_fast_pct  = stat_disagree_ > 0
                         ? 100.0 * stat_disagree_fast_right_ / stat_disagree_ : 0.0;

    printf("\n-------- Router / Fast-Path Statistics --------\n");
    printf("  Total cond-br predictions : %lu\n",           stat_total_);
    printf("  Fast path used            : %lu  (%.2f%%)\n", stat_fast_used_, pct_fast);
    printf("  TAGE path used            : %lu  (%.2f%%)\n", tage_used,       pct_tage);
    printf("  Accuracy when fast used   : %.2f%%\n",        fast_acc);
    printf("  Accuracy when TAGE used   : %.2f%%\n",        tage_acc);
    printf("  Disagreements             : %lu  (%.2f%% of all preds)\n",
           stat_disagree_, pct_disagr);
    printf("  Disagree -> TAGE right    : %lu  (%.2f%% of disagrees)\n",
           stat_disagree_tage_right_, dis_tage_pct);
    printf("  Disagree -> fast right    : %lu  (%.2f%% of disagrees)\n",
           stat_disagree_fast_right_, dis_fast_pct);
    printf("-----------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// Predictor interface
// ---------------------------------------------------------------------------

bool RoutedPredictor::predict(uint64_t seq_no, uint8_t piece,
                               uint64_t pc, bool tage_pred)
{
    bool fp       = gshare_pred_now(pc);
    bool use_tage = router_invoke(pc);

    ++stat_total_;
    if (!use_tage) ++stat_fast_used_;

    pred_time_histories_.emplace(uid(seq_no, piece),
                                 RoutedHist{fast_ghr_, fp, tage_pred, !use_tage});
    return use_tage ? tage_pred : fp;
}

void RoutedPredictor::history_update(uint64_t /*seq_no*/, uint8_t /*piece*/,
                                      uint64_t /*pc*/, bool taken,
                                      uint64_t /*next_pc*/)
{
    fast_ghr_ = ((fast_ghr_ << 1) | (taken ? 1u : 0u)) & ((1u << SHLEN) - 1);
}

void RoutedPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t pc,
                              bool resolveDir, bool /*predDir*/,
                              uint64_t /*nextPC*/)
{
    auto it = pred_time_histories_.find(uid(seq_no, piece));
    if (it == pred_time_histories_.end()) return;
    const RoutedHist& h = it->second;

    int bm_idx = (int)((pc >> 2)           & ((1u << BLOG) - 1));
    int gs_idx = (int)(((pc >> 2) ^ h.ghr) & ((1u << SLOG) - 1));

    // 2-bit bimodal
    bimodal_[bm_idx] = resolveDir ? sat_inc(bimodal_[bm_idx], 3)
                                  : sat_dec(bimodal_[bm_idx], 0);

    // 3-bit gshare (predict-time GHR index)
    gshare_[gs_idx] = resolveDir ? sat_inc(gshare_[gs_idx], 7)
                                 : sat_dec(gshare_[gs_idx], 0);

    // Perceptron router: delta-rule update only on disagreements.
    // Use predict-time h.ghr so indices match exactly what router_invoke read.
    if (h.tage_pred != h.fast_pred) {
        ++stat_disagree_;
        uint32_t mask = (1u << RLOG) - 1;
        int indices[RTABS] = {
            (int)((pc >> 2) & mask),                            // bias
            (int)(((pc >> 2) ^ h.ghr) & mask),                  // PC ^ GHR[9:0]
            (int)(((pc >> 2) ^ (h.ghr >> 5)) & mask),           // PC ^ GHR[14:5]
            (int)(((pc >> 2) ^ (h.ghr ^ (h.ghr >> RLOG))) & mask), // PC ^ fold(GHR)
        };
        if (h.tage_pred == resolveDir) {
            ++stat_disagree_tage_right_;
            for (int t = 0; t < RTABS; ++t)
                router_[t][indices[t]] = sat_inc(router_[t][indices[t]], RWMAX);
        } else {
            ++stat_disagree_fast_right_;
            for (int t = 0; t < RTABS; ++t)
                router_[t][indices[t]] = sat_dec(router_[t][indices[t]], RWMIN);
        }
    }

    // Accuracy per chosen path
    if (h.used_fast) {
        if (h.fast_pred == resolveDir) ++stat_fast_correct_;
    } else {
        if (h.tage_pred == resolveDir) ++stat_tage_correct_;
    }

    pred_time_histories_.erase(it);
}
