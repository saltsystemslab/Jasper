#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "jasper/index/construct_diskann.cuh"

// Directional DiskANN configs: R_int = 32 -> R_fin = 64, k_ranks = M = 4,
// uint8 packing. The differing degree is the interesting part — the PQ payload
// is subspace-major, so its row stride changes between the two.
using int_cfg_t = jasper::graph_construct_config<
    jasper::graph_config<uint32_t, 32, __half, float,
                         jasper::distance_func::L2, true, 4, uint8_t>,
    64, 4, 32, 64>;
using fin_cfg_t = jasper::graph_construct_config<
    jasper::graph_config<uint32_t, 64, __half, float,
                         jasper::distance_func::L2, true, 4, uint8_t>,
    64, 4, 64, 64>;

// Force the whole builder to be compiled for a directional config. Nothing else
// in the tree instantiates one yet (create_diskann_index.cu declares only
// non-directional configs), so without this the entire
// `if constexpr (use_lsh)` half of construct_diskann.cuh — training, the
// populate calls, the payload copy-back — is never type-checked.
template struct jasper::diskann_intermediate_graph<int_cfg_t, fin_cfg_t>;

namespace jasper::test {

using ig_t  = diskann_intermediate_graph<int_cfg_t, fin_cfg_t>;
using int_e = ig_t::int_edge_list_t;
using fin_e = ig_t::fin_edge_list_t;
using int_l = ig_t::int_edge_lsh_t;
using fin_l = ig_t::fin_edge_lsh_t;
using int_q = ig_t::int_edge_pq_t;
using fin_q = ig_t::fin_edge_pq_t;
using int_r = ig_t::int_edge_rq_t;
using fin_r = ig_t::fin_edge_rq_t;

constexpr uint32_t K = 4;   // k_ranks
constexpr uint32_t M = 4;   // pq_m

namespace {

// Give edge slot `e` of an LSH row a value fingerprint derived from `tag`, so a
// misrouted payload is identifiable rather than just "wrong".
template <typename ROW_T>
void stamp_lsh(ROW_T& row, uint8_t e, uint8_t tag) {
  for (uint32_t r = 0; r < K; r++)
    row.packed[static_cast<uint32_t>(e) * K + r] = static_cast<uint8_t>(tag + r);
  row.mag_sq[e] = __float2bfloat16(static_cast<float>(tag));
}

template <typename ROW_T>
void expect_lsh(const ROW_T& row, uint8_t e, uint8_t tag, const char* what) {
  for (uint32_t r = 0; r < K; r++)
    EXPECT_EQ(row.packed[static_cast<uint32_t>(e) * K + r],
              static_cast<uint8_t>(tag + r))
        << what << ": packed[edge " << int(e) << ", rank " << r << "]";
  EXPECT_FLOAT_EQ(__bfloat162float(row.mag_sq[e]), static_cast<float>(tag))
      << what << ": mag_sq[edge " << int(e) << "]";
}

template <typename ROW_T>
void stamp_pq(ROW_T& row, uint8_t e, uint8_t tag) {
  for (uint32_t j = 0; j < M; j++)
    row.set_code(e, j, static_cast<uint8_t>(tag + j));
}

template <typename ROW_T>
void expect_pq(const ROW_T& row, uint8_t e, uint8_t tag, const char* what) {
  for (uint32_t j = 0; j < M; j++)
    EXPECT_EQ(row.get_code(e, j), static_cast<uint8_t>(tag + j))
        << what << ": code[edge " << int(e) << ", sub " << j << "]";
}

// RaBitQ rows are word-plane-major (code[w*N + e]), so N changes with the
// degree — get_word/set_word are what make the R_INT -> R_FIN copy correct, and
// a raw byte copy would scramble every plane past the first. These stamps are
// what would catch that.
template <typename ROW_T>
void stamp_rq(ROW_T& row, uint8_t e, uint8_t tag) {
  for (uint32_t w = 0; w < ROW_T::n_words; w++)
    row.set_word(e, w, 0xA5A50000u | (static_cast<uint32_t>(tag) + w));
  row.rescale[e] = __float2half(static_cast<float>(tag) * 0.25f);
  row.mag_sq[e]  = __float2bfloat16(static_cast<float>(tag));
}

template <typename ROW_T>
void expect_rq(const ROW_T& row, uint8_t e, uint8_t tag, const char* what) {
  for (uint32_t w = 0; w < ROW_T::n_words; w++)
    EXPECT_EQ(row.get_word(e, w), 0xA5A50000u | (static_cast<uint32_t>(tag) + w))
        << what << ": code word[edge " << int(e) << ", w " << w << "]";
  EXPECT_FLOAT_EQ(row.get_rescale(e), static_cast<float>(tag) * 0.25f)
      << what << ": rescale[edge " << int(e) << "]";
  EXPECT_FLOAT_EQ(row.get_mag_sq(e), static_cast<float>(tag))
      << what << ": mag_sq[edge " << int(e) << "]";
}

void set_edge(fin_e& row, uint8_t slot, uint32_t id, float dist) {
  row.edges[slot] = id;
  row.dist[slot]  = __float2bfloat16(dist);
}
void set_edge(int_e& row, uint8_t slot, uint32_t id, float dist) {
  row.edges[slot] = id;
  row.dist[slot]  = __float2bfloat16(dist);
}

}  // namespace

// Baseline: the distance-only merge itself (union, sort, dedup) is unchanged by
// the payload plumbing. Duplicate ids keep the smaller distance.
TEST(DiskannMergeRow, UnionSortsAndDedups) {
  fin_e out{};
  int_e in{};
  uint8_t out_count = 2, in_count = 2;
  set_edge(out, 0, 10, 1.0f);
  set_edge(out, 1, 11, 3.0f);
  set_edge(in,  0, 12, 2.0f);
  set_edge(in,  1, 10, 0.5f);   // duplicate of out[0], closer

  ig_t::merge_row(out, out_count, in, in_count);

  ASSERT_EQ(out_count, 3);
  EXPECT_EQ(out.edges[0], 10u);
  EXPECT_FLOAT_EQ(__bfloat162float(out.dist[0]), 0.5f);
  EXPECT_EQ(out.edges[1], 12u);
  EXPECT_EQ(out.edges[2], 11u);
}

// The aliasing case the payload snapshot exists for: every incoming edge is
// closer than every existing one, so slots 0..1 are overwritten from `in` while
// the `out` payloads that belong in slots 2..3 still live there. Writing the
// payload in place would hand slots 2..3 the values just written to 0..1.
TEST(DiskannMergeRow, PayloadSurvivesInPlaceOverwrite) {
  fin_e out{};
  int_e in{};
  fin_l out_lsh{};
  int_l in_lsh{};
  fin_q out_pq{};
  int_q in_pq{};
  fin_r out_rq{};
  int_r in_rq{};
  uint8_t out_count = 2, in_count = 2;

  set_edge(out, 0, 100, 5.0f);  stamp_lsh(out_lsh, 0, 50); stamp_pq(out_pq, 0, 50); stamp_rq(out_rq, 0, 50);
  set_edge(out, 1, 101, 6.0f);  stamp_lsh(out_lsh, 1, 60); stamp_pq(out_pq, 1, 60); stamp_rq(out_rq, 1, 60);
  set_edge(in,  0, 200, 1.0f);  stamp_lsh(in_lsh,  0, 20); stamp_pq(in_pq,  0, 20); stamp_rq(in_rq,  0, 20);
  set_edge(in,  1, 201, 2.0f);  stamp_lsh(in_lsh,  1, 30); stamp_pq(in_pq,  1, 30); stamp_rq(in_rq,  1, 30);

  ig_t::row_payload_t pay;
  pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;
  pay.out_pq  = &out_pq;  pay.in_pq  = &in_pq;
  pay.out_rq  = &out_rq;  pay.in_rq  = &in_rq;

  ig_t::merge_row(out, out_count, in, in_count, pay);

  ASSERT_EQ(out_count, 4);
  EXPECT_EQ(out.edges[0], 200u);
  EXPECT_EQ(out.edges[1], 201u);
  EXPECT_EQ(out.edges[2], 100u);
  EXPECT_EQ(out.edges[3], 101u);

  // Payloads must have followed their edges, including the two that were read
  // from slots already overwritten.
  expect_lsh(out_lsh, 0, 20, "slot0 from in[0]");
  expect_lsh(out_lsh, 1, 30, "slot1 from in[1]");
  expect_lsh(out_lsh, 2, 50, "slot2 from clobbered out[0]");
  expect_lsh(out_lsh, 3, 60, "slot3 from clobbered out[1]");

  expect_pq(out_pq, 0, 20, "slot0 from in[0]");
  expect_pq(out_pq, 1, 30, "slot1 from in[1]");
  expect_pq(out_pq, 2, 50, "slot2 from clobbered out[0]");
  expect_pq(out_pq, 3, 60, "slot3 from clobbered out[1]");

  expect_rq(out_rq, 0, 20, "slot0 from in[0]");
  expect_rq(out_rq, 1, 30, "slot1 from in[1]");
  expect_rq(out_rq, 2, 50, "slot2 from clobbered out[0]");
  expect_rq(out_rq, 3, 60, "slot3 from clobbered out[1]");
}

// A deduplicated edge keeps the payload of the occurrence that won on distance.
TEST(DiskannMergeRow, DedupKeepsWinningPayload) {
  fin_e out{};
  int_e in{};
  fin_l out_lsh{};
  int_l in_lsh{};
  uint8_t out_count = 1, in_count = 1;

  set_edge(out, 0, 77, 9.0f);  stamp_lsh(out_lsh, 0, 90);
  set_edge(in,  0, 77, 4.0f);  stamp_lsh(in_lsh,  0, 40);   // same id, closer

  ig_t::row_payload_t pay;
  pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;

  ig_t::merge_row(out, out_count, in, in_count, pay);

  ASSERT_EQ(out_count, 1);
  EXPECT_EQ(out.edges[0], 77u);
  EXPECT_FLOAT_EQ(__bfloat162float(out.dist[0]), 4.0f);
  expect_lsh(out_lsh, 0, 40, "kept the closer occurrence's payload");
}

// Truncation to R_FIN, and the invariant the un-cleared payload tail relies on:
// out_count never shrinks, because every surviving `out` id is also a candidate.
TEST(DiskannMergeRow, TruncatesToRFinAndNeverShrinks) {
  constexpr uint32_t R_FIN = ig_t::R_FIN;
  constexpr uint32_t R_INT = ig_t::R_INT;

  fin_e out{};
  int_e in{};
  fin_l out_lsh{};
  int_l in_lsh{};
  fin_r out_rq{};
  int_r in_rq{};
  uint8_t out_count = static_cast<uint8_t>(R_FIN);
  uint8_t in_count  = static_cast<uint8_t>(R_INT);

  // `out` is full and far; `in` is closer, so it takes the front R_INT slots
  // and pushes out the R_INT farthest existing edges.
  for (uint32_t j = 0; j < R_FIN; j++) {
    set_edge(out, static_cast<uint8_t>(j), 1000 + j, 100.0f + j);
    stamp_lsh(out_lsh, static_cast<uint8_t>(j), static_cast<uint8_t>(j));
    stamp_rq(out_rq, static_cast<uint8_t>(j), static_cast<uint8_t>(j));
  }
  for (uint32_t j = 0; j < R_INT; j++) {
    set_edge(in, static_cast<uint8_t>(j), 2000 + j, 1.0f + j);
    stamp_lsh(in_lsh, static_cast<uint8_t>(j), static_cast<uint8_t>(128 + j));
    stamp_rq(in_rq, static_cast<uint8_t>(j), static_cast<uint8_t>(128 + j));
  }

  const uint8_t before = out_count;
  ig_t::row_payload_t pay;
  pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;
  pay.out_rq  = &out_rq;  pay.in_rq  = &in_rq;

  ig_t::merge_row(out, out_count, in, in_count, pay);

  EXPECT_EQ(out_count, static_cast<uint8_t>(R_FIN));
  EXPECT_GE(out_count, before) << "out_count must never shrink";

  for (uint32_t j = 0; j < R_INT; j++) {
    EXPECT_EQ(out.edges[j], 2000 + j);
    expect_lsh(out_lsh, static_cast<uint8_t>(j),
               static_cast<uint8_t>(128 + j), "front slots come from in");
    expect_rq(out_rq, static_cast<uint8_t>(j),
              static_cast<uint8_t>(128 + j), "front slots come from in");
  }
  for (uint32_t j = 0; j < R_FIN - R_INT; j++) {
    EXPECT_EQ(out.edges[R_INT + j], 1000 + j);
    expect_lsh(out_lsh, static_cast<uint8_t>(R_INT + j),
               static_cast<uint8_t>(j), "tail slots are surviving out edges");
    expect_rq(out_rq, static_cast<uint8_t>(R_INT + j),
              static_cast<uint8_t>(j), "tail slots are surviving out edges");
  }
}

// An empty final row (the first shard to touch a node) takes the shard row
// wholesale, payload included.
TEST(DiskannMergeRow, EmptyOutTakesShardRow) {
  fin_e out{};
  int_e in{};
  fin_l out_lsh{};
  int_l in_lsh{};
  fin_q out_pq{};
  int_q in_pq{};
  fin_r out_rq{};
  int_r in_rq{};
  uint8_t out_count = 0, in_count = 3;

  for (uint8_t j = 0; j < 3; j++) {
    set_edge(in, j, 500u + j, 1.0f + j);
    stamp_lsh(in_lsh, j, static_cast<uint8_t>(10 * (j + 1)));
    stamp_pq(in_pq, j, static_cast<uint8_t>(10 * (j + 1)));
    stamp_rq(in_rq, j, static_cast<uint8_t>(10 * (j + 1)));
  }

  ig_t::row_payload_t pay;
  pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;
  pay.out_pq  = &out_pq;  pay.in_pq  = &in_pq;
  pay.out_rq  = &out_rq;  pay.in_rq  = &in_rq;

  ig_t::merge_row(out, out_count, in, in_count, pay);

  ASSERT_EQ(out_count, 3);
  for (uint8_t j = 0; j < 3; j++) {
    EXPECT_EQ(out.edges[j], 500u + j);
    expect_lsh(out_lsh, j, static_cast<uint8_t>(10 * (j + 1)), "copied from in");
    expect_pq(out_pq, j, static_cast<uint8_t>(10 * (j + 1)), "copied from in");
    expect_rq(out_rq, j, static_cast<uint8_t>(10 * (j + 1)), "copied from in");
  }
  // Slots past the count are untouched, which is what lets the merge skip
  // clearing them (the arrays start value-initialized).
  for (uint32_t j = 3; j < ig_t::R_FIN; j++) {
    EXPECT_FLOAT_EQ(__bfloat162float(out_lsh.mag_sq[j]), 0.0f);
    for (uint32_t r = 0; r < K; r++)
      EXPECT_EQ(out_lsh.packed[j * K + r], 0u);
  }
}

// INVALID neighbor slots inside the count are dropped, and don't consume an
// output slot or shift the payload mapping.
TEST(DiskannMergeRow, SkipsInvalidIds) {
  constexpr uint32_t INVALID = std::numeric_limits<uint32_t>::max();
  fin_e out{};
  int_e in{};
  fin_l out_lsh{};
  int_l in_lsh{};
  uint8_t out_count = 0, in_count = 3;

  set_edge(in, 0, INVALID, 0.1f); stamp_lsh(in_lsh, 0, 11);
  set_edge(in, 1, 42,      2.0f); stamp_lsh(in_lsh, 1, 22);
  set_edge(in, 2, 43,      3.0f); stamp_lsh(in_lsh, 2, 33);

  ig_t::row_payload_t pay;
  pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;

  ig_t::merge_row(out, out_count, in, in_count, pay);

  ASSERT_EQ(out_count, 2);
  EXPECT_EQ(out.edges[0], 42u);
  EXPECT_EQ(out.edges[1], 43u);
  expect_lsh(out_lsh, 0, 22, "payload tracks the surviving in slot");
  expect_lsh(out_lsh, 1, 33, "payload tracks the surviving in slot");
}

// Repeated merges (a node duplicated into several shards) accumulate correctly:
// payload identity is preserved across more than one pass.
TEST(DiskannMergeRow, RepeatedMergesAccumulate) {
  fin_e out{};
  fin_l out_lsh{};
  uint8_t out_count = 0;

  for (uint8_t shard = 0; shard < 3; shard++) {
    int_e in{};
    int_l in_lsh{};
    for (uint8_t j = 0; j < 2; j++) {
      const uint32_t id = 10u * shard + j;
      set_edge(in, j, id, static_cast<float>(id + 1));
      stamp_lsh(in_lsh, j, static_cast<uint8_t>(id + 1));
    }
    ig_t::row_payload_t pay;
    pay.out_lsh = &out_lsh; pay.in_lsh = &in_lsh;
    ig_t::merge_row(out, out_count, in, 2, pay);
  }

  ASSERT_EQ(out_count, 6);
  for (uint8_t s = 0; s < out_count; s++) {
    const uint32_t id = out.edges[s];
    expect_lsh(out_lsh, s, static_cast<uint8_t>(id + 1),
               "each edge still carries its own codes");
  }
}

}  // namespace jasper::test
