#pragma once

namespace jasper {

struct construct_timer {
  float beam_search_ms   = 0;
  float merge_cands_ms   = 0;
  float robust_prune_ms  = 0;
  float fill_reverse_ms  = 0;
  float sort_offsets_ms  = 0;
  float reverse_prune_ms = 0;
  uint32_t n_batches     = 0;

  void print() const {
    float total = beam_search_ms + merge_cands_ms + robust_prune_ms
                + fill_reverse_ms + sort_offsets_ms + reverse_prune_ms;
    std::printf("\n[construct] timing summary (%u batches, %.1f ms total)\n", n_batches, total);
    std::printf("  beam search      : %8.1f ms  (%5.1f%%)\n", beam_search_ms,   100.f * beam_search_ms / total);
    std::printf("  merge candidates : %8.1f ms  (%5.1f%%)\n", merge_cands_ms,   100.f * merge_cands_ms / total);
    std::printf("  robust prune     : %8.1f ms  (%5.1f%%)\n", robust_prune_ms,  100.f * robust_prune_ms / total);
    std::printf("  fill reverse     : %8.1f ms  (%5.1f%%)\n", fill_reverse_ms,  100.f * fill_reverse_ms / total);
    std::printf("  sort + offsets   : %8.1f ms  (%5.1f%%)\n", sort_offsets_ms,  100.f * sort_offsets_ms / total);
    std::printf("  reverse prune    : %8.1f ms  (%5.1f%%)\n", reverse_prune_ms, 100.f * reverse_prune_ms / total);
  }
};

inline float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  return ms;
}

}