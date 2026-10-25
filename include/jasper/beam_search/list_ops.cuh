// jasper/beam_search/list_ops.cuh
//
// Cheaper candidate-list maintenance for graph_beam_search_kernel, one
// list_opt bit (config.cuh) each. The reference path, per pop, after u's
// neighbors are appended and estimated:
//
//   6b  a duplicate check marks copies of listed nodes DROPPED (new or old),
//   7   a CUB block sort orders the new entries and a merge-path merge joins
//       them to the list,
//   8   dedup_results removes the DROPPED entries, and
//   9   clip_k cuts the list back to its capacity.
//
// Each step touches the whole list and costs several block barriers; most of
// a pop's time goes there (sbatch/profile_search.sbatch). The variants:
//
//   LIST_PREFILTER    once the list is full, a new entry ranked after its
//                     last entry T cannot survive the clip: every listed entry
//                     ranked before it either stays or is dropped in favour of
//                     a better new copy, which then ranks before it too. (That
//                     copy is never itself dropped for being in the top-k: a
//                     top-k node's list entry is visited or gone, and 6b does
//                     not re-add a top-k node.) So it is dropped before 6b, and
//                     a pop that keeps none skips 6b-9 outright.
//   LIST_RANK_SORT    the new entries that are not DROPPED are gathered and
//                     sorted by rank, one thread each (rank_sort_new_entries),
//                     instead of a CUB block sort over all 64 slots, then
//                     merged by the reference merge.
//   LIST_FUSED_MERGE  8 and 9 fold into the merge: every entry goes straight
//                     to its final position, list entry i to i - (old entries
//                     dropped before it) + (new entries ranked before it), new
//                     entry j to j + (live list entries ranked before it), and
//                     only the part of the list from the first change on is
//                     moved. 6b records dropped old entries by position rather
//                     than overwriting their ids, which keeps the list sorted.
//                     (The reference overwrites the id with DROPPED_INDEX, and
//                     with equal estimates that can unsort it, so its merge
//                     can misorder a new entry against a tied list entry; only
//                     there can the two differ.)
//
// Every variant leaves the list as the reference does after each pop, so a
// search with any list_opts makes the same pops and returns the same results;
// distortion_test --check_list_opts compares them query by query.
#pragma once

#include <cstdint>

#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"  // CustomPairLess, merge_tail_into_head

namespace jasper {

// Index of empty_entry() and of an out-of-range neighbor's (visited) entry.
static constexpr uint32_t LIST_INVALID_INDEX = 0x7FFFFFFFu;

// An out-of-range neighbor's entry: kept in the list as a visited placeholder.
__device__ __forceinline__ bool is_invalid_slot(ENTRY_T x) {
  return get_index(x) == LIST_INVALID_INDEX && get_visited(x);
}

// Gathers the new entries result_buffer[offset .. offset + n_new) that are
// not DROPPED into scratch and sorts them by rank: each, one thread per entry,
// counts the entries ranked before it (ties, which only identical out-of-range
// placeholders can make, broken by slot) and moves to that place. With the
// prefilter on, 6-12 of 64 survive a typical pop, so this is a handful of
// compares per thread, all at once. (A 64-wide bitonic network in one warp was
// tried first: 21 dependent shuffle steps while the block waits, slower than
// the CUB block sort it replaced.)
//
// With dedup_out_of_range it also does what dedup_results does to the
// placeholders, which all compare equal: at most one stays in the list, and
// none is added when the list (result_buffer[0 .. offset)) already ends with
// one. If drops is given, threads MAX_NEIGHBORS .. sort the n_drops dropped
// positions the same way at the same time. Every thread must call it; returns
// the survivor count.
template <uint32_t BLOCK_SIZE, uint32_t MAX_NEIGHBORS>
__device__ uint32_t rank_sort_new_entries(const ENTRY_T* __restrict__ result_buffer,
                                          uint32_t offset, uint32_t n_new,
                                          ENTRY_T* __restrict__ scratch,
                                          bool dedup_out_of_range,
                                          uint32_t* __restrict__ drops, uint32_t n_drops) {
  static_assert(BLOCK_SIZE >= 2 * MAX_NEIGHBORS,
                "one thread per new entry, and one per dropped position");
  __shared__ uint32_t s_n, s_placeholder;
  if (threadIdx.x == 0) {
    s_n = 0;
    s_placeholder = dedup_out_of_range && offset > 0 &&
                    get_index(result_buffer[offset - 1]) == LIST_INVALID_INDEX;
  }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < n_new; j += BLOCK_SIZE) {
    const ENTRY_T e = result_buffer[offset + j];
    if (get_index(e) == DROPPED_INDEX) continue;
    if (dedup_out_of_range && is_invalid_slot(e) && atomicExch(&s_placeholder, 1u) != 0)
      continue;
    scratch[atomicAdd(&s_n, 1u)] = e;
  }
  __syncthreads();

  const uint32_t n = s_n;
  const uint32_t t = threadIdx.x;
  const bool sorts_entry = t < n;
  const bool sorts_drop  = drops != nullptr && t >= MAX_NEIGHBORS && t - MAX_NEIGHBORS < n_drops;
  CustomPairLess less;
  ENTRY_T  e = 0;
  uint32_t d = 0, rank = 0;
  if (sorts_entry) {
    e = scratch[t];
    for (uint32_t m = 0; m < n; ++m) {
      const ENTRY_T o = scratch[m];
      rank += less(o, e) || (m < t && !less(e, o));
    }
  } else if (sorts_drop) {
    d = drops[t - MAX_NEIGHBORS];
    for (uint32_t m = 0; m < n_drops; ++m) rank += drops[m] < d;  // positions are distinct
  }
  __syncthreads();
  if (sorts_entry)     scratch[rank] = e;
  else if (sorts_drop) drops[rank]   = d;
  __syncthreads();
  return n;
}

// LIST_FUSED_MERGE: the list is result_buffer[0 .. head_len), sorted, with the
// n_drops entries at positions drops[] (sorted) to be removed; survivors[0 ..
// n_surv) are the new entries to add, sorted, none tied with a list entry.
// Moves every entry straight to its place in the merged, deduplicated list,
// keeps the first `capacity`, and sets the count. Entries before the first
// change (the first drop, or where the best new entry goes) stay put.
template <uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH, uint32_t MAX_NEIGHBORS>
__device__ void fused_merge_clip(ENTRY_T* __restrict__ result_buffer,
                                 uint32_t* __restrict__ result_buffer_count,
                                 uint32_t head_len,
                                 const ENTRY_T* __restrict__ survivors, uint32_t n_surv,
                                 const uint32_t* __restrict__ drops, uint32_t n_drops,
                                 uint32_t capacity) {
  CustomPairLess less;
  // Dropped positions below i.
  const auto drops_before = [&](uint32_t i) {
    uint32_t lo = 0, hi = n_drops;
    while (lo < hi) { const uint32_t m = (lo + hi) >> 1; if (drops[m] < i) lo = m + 1; else hi = m; }
    return lo;
  };
  // New entries ranked before e.
  const auto survivors_before = [&](ENTRY_T e) {
    uint32_t lo = 0, hi = n_surv;
    while (lo < hi) { const uint32_t m = (lo + hi) >> 1; if (less(survivors[m], e)) lo = m + 1; else hi = m; }
    return lo;
  };
  // List entries (dropped ones included) ranked before e.
  const auto head_before = [&](ENTRY_T e) {
    uint32_t lo = 0, hi = head_len;
    while (lo < hi) { const uint32_t m = (lo + hi) >> 1; if (less(result_buffer[m], e)) lo = m + 1; else hi = m; }
    return lo;
  };

  uint32_t first = head_len;
  if (n_drops > 0) first = drops[0];
  if (n_surv > 0)  first = min(first, head_before(survivors[0]));

  // Read everything that moves before anything is written.
  constexpr uint32_t EPT = (MAX_SEARCH_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE;
  constexpr uint32_t SPT = (MAX_NEIGHBORS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  ENTRY_T  a[EPT], b[SPT];
  uint32_t a_to[EPT], b_to[SPT];
  #pragma unroll
  for (uint32_t t = 0; t < EPT; ++t) {
    const uint32_t i = first + threadIdx.x + t * BLOCK_SIZE;
    a_to[t] = 0xFFFFFFFFu;
    if (i < head_len) {
      const uint32_t db = drops_before(i);
      if (db == n_drops || drops[db] != i) {
        a[t]    = result_buffer[i];
        a_to[t] = i - db + survivors_before(a[t]);
      }
    }
  }
  #pragma unroll
  for (uint32_t t = 0; t < SPT; ++t) {
    const uint32_t j = threadIdx.x + t * BLOCK_SIZE;
    b_to[t] = 0xFFFFFFFFu;
    if (j < n_surv) {
      b[t] = survivors[j];
      const uint32_t lb = head_before(b[t]);
      b_to[t] = j + lb - drops_before(lb);
    }
  }
  __syncthreads();
  #pragma unroll
  for (uint32_t t = 0; t < EPT; ++t) if (a_to[t] < capacity) result_buffer[a_to[t]] = a[t];
  #pragma unroll
  for (uint32_t t = 0; t < SPT; ++t) if (b_to[t] < capacity) result_buffer[b_to[t]] = b[t];
  if (threadIdx.x == 0)
    result_buffer_count[0] = min(capacity, head_len - n_drops + n_surv);
  __syncthreads();
}

}  // namespace jasper
