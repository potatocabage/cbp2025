#ifndef _ROUTED_PREDICTOR_H_
#define _ROUTED_PREDICTOR_H_

// ================================================================
// Routed two-level predictor
//
// Fast path : bimodal (2-bit, PC-indexed) +
//             gshare  (3-bit, PC ^ 15-bit GHR indexed)
//             — always runs, always updates
//
// Full path : TAGE-SC-L prediction passed in from the interface
//             — prediction output gated by the router
//
// Router    : hashed perceptron — RTABS tables, each indexed by a different
//             hash of (PC, GHR); decision = sign(sum of weights) > RTHETA
//               sum > RTHETA  →  invoke TAGE-SC-L
//               sum <= RTHETA →  use fast-path gshare prediction
//
//             Table hashes (all masked to RLOG bits):
//               t=0  (PC >> 2)                         — per-branch bias
//               t=1  (PC >> 2) ^ GHR[9:0]             — PC × short history
//               t=2  (PC >> 2) ^ (GHR >> 5)           — PC × upper GHR bits
//               t=3  (PC >> 2) ^ fold(GHR, RLOG)      — PC × full folded GHR
//
// Online learning rule (applied at resolve):
//   tage_pred != fast_pred AND tage correct  →  all weights += 1 (reinforce invoke)
//   tage_pred != fast_pred AND fast correct  →  all weights -= 1 (reinforce skip)
//   tage_pred == fast_pred                   →  no update (agreed; no routing signal)
//
// Storage budget (logical bits, not counting history checkpoints):
//   bimodal  2^14 * 2 bits =  32 Kbits  ( 4 KB)
//   gshare   2^13 * 3 bits =  24 Kbits  ( 3 KB)
//   router   RTABS * 2^RLOG * 8 bits = 32 Kbits  ( 4 KB)
//   Total                            = 88 Kbits  (11 KB) — well within 128 KB budget
// ================================================================

#include <cassert>
#include <cstdint>
#include <unordered_map>

static constexpr int BLOG  = 14;   // log2 bimodal entries
static constexpr int SLOG  = 13;   // log2 gshare  entries
static constexpr int RLOG  = 10;   // log2 entries per router perceptron table
static constexpr int RTABS = 4;    // number of router perceptron tables
static constexpr int RTHETA = 0;   // invoke TAGE if perceptron sum > RTHETA
static constexpr int8_t RWMAX =  15;  // weight saturation bounds
static constexpr int8_t RWMIN = -15;
static constexpr int SHLEN = 15;   // short GHR length in bits

// Predict-time state checkpointed per conditional branch
struct RoutedHist {
    uint32_t ghr;       // fast-path GHR at predict time
    bool     fast_pred; // gshare prediction
    bool     tage_pred; // TAGE-SC-L prediction
    bool     used_fast; // true if router chose the fast path
};

class RoutedPredictor {
    // --- prediction tables ---
    int8_t bimodal_[1 << BLOG];  // 2-bit counters {0..3}; taken >= 2
    int8_t gshare_ [1 << SLOG];  // 3-bit counters {0..7}; taken >= 4
    int8_t router_[RTABS][1 << RLOG];  // perceptron weights; sum > RTHETA → invoke TAGE

    uint32_t fast_ghr_;          // speculative short GHR

    // --- stats ---
    uint64_t stat_total_;
    uint64_t stat_fast_used_;
    uint64_t stat_fast_correct_;
    uint64_t stat_tage_correct_;
    uint64_t stat_disagree_;
    uint64_t stat_disagree_tage_right_;
    uint64_t stat_disagree_fast_right_;

    std::unordered_map<uint64_t, RoutedHist> pred_time_histories_;

    // Trivial helpers kept inline for call-site optimisation
    static uint64_t uid(uint64_t seq_no, uint8_t piece) {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0xFu);
    }
    static int8_t sat_inc(int8_t c, int8_t mx) { return c < mx ? c + 1 : mx; }
    static int8_t sat_dec(int8_t c, int8_t mn) { return c > mn ? c - 1 : mn; }

    bool gshare_pred_now(uint64_t pc) const;
    bool router_invoke(uint64_t pc) const;

public:
    RoutedPredictor();

    void setup();
    void terminate();

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t pc, bool tage_pred);
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t pc,
                        bool taken, uint64_t next_pc);
    void update(uint64_t seq_no, uint8_t piece, uint64_t pc,
                bool resolveDir, bool predDir, uint64_t nextPC);
};

extern RoutedPredictor cond_predictor_impl;

#endif // _ROUTED_PREDICTOR_H_
