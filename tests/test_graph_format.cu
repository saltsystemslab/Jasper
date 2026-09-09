#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jasper/index/construct_diskann.cuh"
#include "jasper/index/graph.cuh"

namespace jasper::test {

// ── base_section_bytes: where a directional trailer starts ─────────────────
//
// The trailer has no back-pointer, so save/load agree on its offset only if
// base_section_bytes reproduces exactly what a writer emitted. It infers the
// per-node columns — and therefore whether the trailing packed-deletion block
// and next_id field exist — from the header's bytes_per_node alone
// (load_graph_from_file's bpn_* ladder).
//
// These tests pin that inference against the layouts that have shipped. They
// deliberately re-derive each writer's byte count from sizeof(...) instead of
// calling the helper, so a wrong ladder shows up as a mismatch rather than
// two copies of the same mistake. Pure host arithmetic; no device needed.

namespace {

template <uint8_t R>
using cfg_r = graph_config<uint32_t, R, __half, float, distance_func::L2>;

// The four per-node record sizes load_graph_from_file distinguishes.
template <typename cfg>
struct bpn_of {
  using data_t      = typename cfg::data_t;
  using edge_list_t = typename cfg::edge_list_t;
  using index_t     = typename cfg::index_t;

  static uint64_t base(uint32_t dim) {
    return sizeof(data_t) * dim + sizeof(uint8_t) + sizeof(edge_list_t);
  }
  static uint64_t with_deleted(uint32_t dim) { return base(dim) + sizeof(uint8_t); }
  static uint64_t with_id(uint32_t dim)      { return with_deleted(dim) + sizeof(index_t); }
  static uint64_t bitpacked(uint32_t dim)    { return base(dim) + sizeof(index_t); }
};

constexpr uint64_t HEADER = 4 * sizeof(uint64_t);

}  // namespace

template <typename cfg>
static void check_layouts(uint32_t dim, uint64_t n) {
  using L = bpn_of<cfg>;
  const std::string tag =
      " (dim=" + std::to_string(dim) + ", n=" + std::to_string(n) + ")";

  // 1. What save_graph_to_file writes today: slot_to_id inline, deletion as a
  //    packed block after all node records, then a trailing next_id.
  EXPECT_EQ(base_section_bytes<cfg>(dim, n, L::bitpacked(dim)),
            HEADER + n * L::bitpacked(dim) + (n + 7) / 8 + sizeof(uint64_t))
      << "bitpacked+id layout" << tag;

  // 2. What construct_diskann's save_to_file writes: legacy, no trailing blocks.
  EXPECT_EQ(base_section_bytes<cfg>(dim, n, L::base(dim)),
            HEADER + n * L::base(dim))
      << "legacy layout" << tag;

  // 3. Legacy + inline deleted byte: still no trailing blocks.
  EXPECT_EQ(base_section_bytes<cfg>(dim, n, L::with_deleted(dim)),
            HEADER + n * L::with_deleted(dim))
      << "legacy+deleted layout" << tag;

  // 4. Legacy + inline deleted byte + id column: trailing next_id, no block.
  EXPECT_EQ(base_section_bytes<cfg>(dim, n, L::with_id(dim)),
            HEADER + n * L::with_id(dim) + sizeof(uint64_t))
      << "legacy+deleted+id layout" << tag;

  // The ladder is only decidable if the four record sizes stay distinct.
  const uint64_t v[4] = {L::base(dim), L::with_deleted(dim), L::with_id(dim),
                         L::bitpacked(dim)};
  for (int i = 0; i < 4; i++)
    for (int j = i + 1; j < 4; j++)
      EXPECT_NE(v[i], v[j]) << "bytes_per_node collision " << i << "/" << j << tag;
}

TEST(BaseSectionBytes, LayoutsR32) {
  check_layouts<cfg_r<32>>(96, 1);            // single node
  check_layouts<cfg_r<32>>(96, 7);            // n not a multiple of 8 (ceil(n/8))
  check_layouts<cfg_r<32>>(96, 1000000);
  check_layouts<cfg_r<32>>(96, 1000000000ull);  // 1B x d96
}

TEST(BaseSectionBytes, LayoutsR64) {
  check_layouts<cfg_r<64>>(768, 1);
  check_layouts<cfg_r<64>>(768, 7);
  check_layouts<cfg_r<64>>(768, 100000000ull);  // 100M x d768
}

// The offset must be computed in 64-bit throughout: at 1B x d96 the base
// section alone is ~385 GB, which overflows uint32 by two orders of magnitude.
TEST(BaseSectionBytes, NoOverflowAtBillionScale) {
  using cfg = cfg_r<32>;
  const uint64_t n = 1000000000ull;
  const uint64_t bpn = bpn_of<cfg>::bitpacked(96);
  const uint64_t got = base_section_bytes<cfg>(96, n, bpn);
  EXPECT_GT(got, uint64_t{1} << 32);
  EXPECT_EQ(got, HEADER + n * bpn + (n + 7) / 8 + sizeof(uint64_t));
}

// An empty index still has a well-formed base section: just the header.
TEST(BaseSectionBytes, EmptyIndex) {
  using cfg = cfg_r<32>;
  EXPECT_EQ(base_section_bytes<cfg>(96, 0, bpn_of<cfg>::base(96)), HEADER);
  EXPECT_EQ(base_section_bytes<cfg>(96, 0, bpn_of<cfg>::bitpacked(96)),
            HEADER + sizeof(uint64_t));  // next_id is written even for N=0
}

// directional_trailer_offset reads n_vectors/bytes_per_node out of the file's
// own 4-u64 header, and must agree with base_section_bytes on those values.
TEST(DirectionalTrailerOffset, ReadsHeaderFromFile) {
  using cfg = cfg_r<32>;
  const uint32_t dim = 96;
  const uint64_t n   = 12345;
  const uint64_t bpn = bpn_of<cfg>::bitpacked(dim);

  const std::string path =
      std::string(::testing::TempDir()) + "jasper_trailer_offset_test.bin";
  {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    const uint64_t hdr[4] = {/*total_file_size=*/0, n, /*medoid=*/0, bpn};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  }

  EXPECT_EQ(directional_trailer_offset<cfg>(path, dim),
            base_section_bytes<cfg>(dim, n, bpn));
  std::remove(path.c_str());
}

TEST(DirectionalTrailerOffset, ThrowsOnTruncatedHeader) {
  using cfg = cfg_r<32>;
  const std::string path =
      std::string(::testing::TempDir()) + "jasper_trailer_offset_short.bin";
  {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    const uint32_t partial = 0;
    f.write(reinterpret_cast<const char*>(&partial), sizeof(partial));
  }

  EXPECT_THROW(directional_trailer_offset<cfg>(path, 96), std::runtime_error);
  std::remove(path.c_str());
}

TEST(DirectionalTrailerOffset, ThrowsOnMissingFile) {
  using cfg = cfg_r<32>;
  EXPECT_THROW(
      directional_trailer_offset<cfg>("/nonexistent/jasper/no_such_index.bin", 96),
      std::runtime_error);
}

// ── diskann_intermediate_graph::save_directional_to_file byte layout ────────
//
// The trailer has no back-pointer and no length fields: the reader walks it by
// knowing the block order and sizes. This pins that layout at the byte level
// against what load_directional_graph_from_file reads, without needing a device
// — the PQ block is left out precisely because its centroids live in device
// memory, so an LSH+norms trailer is exercisable on host alone.

namespace {

// R_int == R_fin == 32, k_ranks 4, uint8 packing. Deliberately a different
// specialization from the one tests/test_diskann_merge.cu explicitly
// instantiates, so the two TUs can't collide over it.
using fmt_int_cfg = graph_construct_config<
    graph_config<uint32_t, 32, __half, float, distance_func::L2, true, 4, uint8_t>,
    64, 4, 32, 64>;
using fmt_fin_cfg = fmt_int_cfg;
using fmt_ig_t    = diskann_intermediate_graph<fmt_int_cfg, fmt_fin_cfg>;
using fmt_cfg     = fmt_fin_cfg::graph_cfg_t;

}  // namespace

TEST(DiskannDirectionalSave, TrailerLayoutAndOffsets) {
  using edge_lsh_t = fmt_ig_t::fin_edge_lsh_t;

  constexpr uint32_t dim = 8;
  constexpr uint32_t N   = 100;

  // Fake dataset in plain host memory — save_to_file only reads through it, and
  // vector_view over a raw pointer keeps this test CUDA-free.
  std::vector<__half> vecs(static_cast<size_t>(N) * dim);
  for (size_t i = 0; i < vecs.size(); i++)
    vecs[i] = __float2half(static_cast<float>(i % 17));
  fmt_ig_t::vector_view_t view(vecs.data(), dim, N, /*on_host=*/true);
  ASSERT_EQ(view.padded_dim, dim) << "test assumes dim is already 16B-aligned";

  fmt_ig_t ig;
  ig.n_vectors = N;
  ig.dim       = dim;
  ig.edges.assign(N, fmt_ig_t::fin_edge_list_t{});
  ig.counts.assign(N, 0);
  for (uint32_t i = 0; i < N; i++) {
    ig.counts[i] = 3;
    for (uint8_t j = 0; j < 3; j++) ig.edges[i].edges[j] = (i + j) % N;
  }

  // Distinct byte patterns so a misplaced block shows up as a mismatch.
  ig.dir.has_lsh = true;
  ig.dir.edge_lshs.assign(N, edge_lsh_t{});
  for (uint32_t i = 0; i < N; i++)
    for (uint32_t b = 0; b < sizeof(edge_lsh_t::packed); b++)
      ig.dir.edge_lshs[i].packed[b] = static_cast<uint8_t>((i * 31 + b) & 0xFF);
  for (uint32_t r = 0; r < fmt_cfg::k_ranks; r++)
    ig.dir.globals.c_per_rank[r] = 0.5f + static_cast<float>(r);
  ig.dir.globals.norm_denom = 42.0f;

  ig.dir.norms.resize(N);
  for (uint32_t i = 0; i < N; i++) ig.dir.norms[i] = static_cast<float>(i) * 0.25f;

  const std::string path =
      std::string(::testing::TempDir()) + "jasper_diskann_trailer.bin";
  ig.save_directional_to_file(view, path);

  // ── total size ──
  const uint64_t bpn = sizeof(__half) * dim + sizeof(uint8_t)
                     + sizeof(fmt_ig_t::fin_edge_list_t);
  const uint64_t base = 4 * sizeof(uint64_t) + uint64_t{N} * bpn;
  const uint64_t trailer = 2 * sizeof(uint32_t) + sizeof(uint8_t)      // prefix
                         + 2 * sizeof(uint8_t)                          // k_ranks, packed
                         + sizeof(lsh_globals<fmt_cfg::k_ranks>)
                         + uint64_t{N} * sizeof(edge_lsh_t)
                         + uint64_t{N} * sizeof(float);                 // norms
  {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.is_open());
    EXPECT_EQ(static_cast<uint64_t>(f.tellg()), base + trailer);
  }

  // The writer's base section must end exactly where the reader looks.
  EXPECT_EQ(directional_trailer_offset<fmt_cfg>(path, dim), base);

  // ── trailer contents, read the way the loader reads them ──
  std::ifstream f(path, std::ios::binary);
  ASSERT_TRUE(f.is_open());
  f.seekg(static_cast<std::streamoff>(base));

  uint32_t magic = 0, version = 0;
  uint8_t  flags = 0;
  f.read(reinterpret_cast<char*>(&magic),   sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(&flags),   sizeof(uint8_t));
  EXPECT_EQ(magic, DIRECTIONAL_TRAILER_MAGIC);
  EXPECT_EQ(version, DIRECTIONAL_TRAILER_VERSION);
  EXPECT_EQ(flags, 1u | 4u) << "lsh + norms, no pq";

  uint8_t k_ranks_u8 = 0, packed_size_u8 = 0;
  f.read(reinterpret_cast<char*>(&k_ranks_u8),     sizeof(uint8_t));
  f.read(reinterpret_cast<char*>(&packed_size_u8), sizeof(uint8_t));
  EXPECT_EQ(k_ranks_u8, fmt_cfg::k_ranks);
  EXPECT_EQ(packed_size_u8, sizeof(typename fmt_cfg::packed_t));

  lsh_globals<fmt_cfg::k_ranks> globals{};
  f.read(reinterpret_cast<char*>(&globals), sizeof(globals));
  for (uint32_t r = 0; r < fmt_cfg::k_ranks; r++)
    EXPECT_FLOAT_EQ(globals.c_per_rank[r], 0.5f + static_cast<float>(r));
  EXPECT_FLOAT_EQ(globals.norm_denom, 42.0f);

  std::vector<edge_lsh_t> read_lsh(N);
  f.read(reinterpret_cast<char*>(read_lsh.data()),
         static_cast<std::streamsize>(N * sizeof(edge_lsh_t)));
  EXPECT_EQ(std::memcmp(read_lsh.data(), ig.dir.edge_lshs.data(),
                        N * sizeof(edge_lsh_t)), 0)
      << "edge_lshs block round-tripped byte-for-byte";

  std::vector<float> read_norms(N);
  f.read(reinterpret_cast<char*>(read_norms.data()),
         static_cast<std::streamsize>(N * sizeof(float)));
  for (uint32_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(read_norms[i], static_cast<float>(i) * 0.25f);

  EXPECT_TRUE(f.good()) << "stream should not be exhausted early";
  f.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  EXPECT_TRUE(f.eof()) << "nothing should follow the norms block";

  f.close();
  std::remove(path.c_str());
}

// The rotation block (flag bit 3) sits after everything else, so a reader that
// only knows bits 0-2 stops exactly where the norms end and never reads into
// it — that's what makes it additive in both directions instead of a version
// bump, which the reader would reject outright.
TEST(DiskannDirectionalSave, RotationBlockIsLastAndOptional) {
  using edge_lsh_t = fmt_ig_t::fin_edge_lsh_t;

  constexpr uint32_t dim = 8;
  constexpr uint32_t N   = 4;

  std::vector<__half> vecs(static_cast<size_t>(N) * dim, __float2half(0.5f));
  fmt_ig_t::vector_view_t view(vecs.data(), dim, N, /*on_host=*/true);

  auto make = [&](bool record_rotation, bool prerotate, uint64_t seed) {
    fmt_ig_t ig;
    ig.n_vectors = N;
    ig.dim       = dim;
    ig.edges.assign(N, fmt_ig_t::fin_edge_list_t{});
    ig.counts.assign(N, 1);
    ig.dir.has_lsh = true;
    ig.dir.edge_lshs.assign(N, edge_lsh_t{});
    ig.dir.rotation.known     = record_rotation;
    ig.dir.rotation.prerotate = prerotate;
    ig.dir.rotation.seed      = seed;
    return ig;
  };

  const uint64_t bpn = sizeof(__half) * dim + sizeof(uint8_t)
                     + sizeof(fmt_ig_t::fin_edge_list_t);
  const uint64_t base = 4 * sizeof(uint64_t) + uint64_t{N} * bpn;
  const uint64_t rot_offset = base
      + 2 * sizeof(uint32_t) + sizeof(uint8_t)
      + 2 * sizeof(uint8_t) + sizeof(lsh_globals<fmt_cfg::k_ranks>)
      + uint64_t{N} * sizeof(edge_lsh_t);
  const uint64_t rot_block = sizeof(uint8_t) + sizeof(uint64_t);

  const std::string path =
      std::string(::testing::TempDir()) + "jasper_diskann_rot.bin";

  // With the block: flags carry bit 3, and it occupies the final 9 bytes.
  {
    auto ig = make(/*record_rotation=*/true, /*prerotate=*/true, 0xDEADBEEFull);
    ig.save_directional_to_file(view, path);

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.is_open());
    EXPECT_EQ(static_cast<uint64_t>(f.tellg()), rot_offset + rot_block);

    f.seekg(static_cast<std::streamoff>(base + 2 * sizeof(uint32_t)));
    uint8_t flags = 0;
    f.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    EXPECT_EQ(flags, 1u | 8u) << "lsh + rotation";

    f.seekg(static_cast<std::streamoff>(rot_offset));
    uint8_t  rot  = 0;
    uint64_t seed = 0;
    f.read(reinterpret_cast<char*>(&rot),  sizeof(uint8_t));
    f.read(reinterpret_cast<char*>(&seed), sizeof(uint64_t));
    EXPECT_EQ(rot, 1u);
    EXPECT_EQ(seed, 0xDEADBEEFull) << "seed must round-trip at full 64 bits";
    f.close();
  }

  // "Known and NOT prerotated" is a distinct, recorded state — a definitive
  // instruction not to rotate queries, not the same as saying nothing.
  {
    auto ig = make(/*record_rotation=*/true, /*prerotate=*/false, 0);
    ig.save_directional_to_file(view, path);

    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.seekg(static_cast<std::streamoff>(base + 2 * sizeof(uint32_t)));
    uint8_t flags = 0;
    f.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    EXPECT_EQ(flags, 1u | 8u) << "bit 3 set even when prerotate is false";

    f.seekg(static_cast<std::streamoff>(rot_offset));
    uint8_t rot = 0xFF;
    f.read(reinterpret_cast<char*>(&rot), sizeof(uint8_t));
    EXPECT_EQ(rot, 0u);
    f.close();
  }

  // Without the block: bit 3 clear and the file ends where the norms would,
  // i.e. byte-identical to what a pre-rotation-block writer produced.
  {
    auto ig = make(/*record_rotation=*/false, /*prerotate=*/false, 0);
    ig.save_directional_to_file(view, path);

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(f.is_open());
    EXPECT_EQ(static_cast<uint64_t>(f.tellg()), rot_offset);

    f.seekg(static_cast<std::streamoff>(base + 2 * sizeof(uint32_t)));
    uint8_t flags = 0;
    f.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    EXPECT_EQ(flags, 1u);
    f.close();
  }

  std::remove(path.c_str());
}

// read_rotation_info exists so a NON-directional config can find out which
// space an index is in — it can't call load_directional_graph_from_file, and
// couldn't size the estimator blocks anyway. So read it here with cfg_r<32>,
// which has no LSH storage at all but the same per-node record size, and check
// it agrees with what the directional writer put there.
TEST(ReadRotationInfo, ReadableFromANonDirectionalConfig) {
  using plain_cfg  = cfg_r<32>;   // use_lsh == false
  using edge_lsh_t = fmt_ig_t::fin_edge_lsh_t;

  constexpr uint32_t dim = 8;
  constexpr uint32_t N   = 6;
  static_assert(sizeof(plain_cfg::edge_list_t) == sizeof(fmt_cfg::edge_list_t),
                "test relies on both configs sharing a per-node record size");

  std::vector<__half> vecs(static_cast<size_t>(N) * dim, __float2half(0.25f));
  fmt_ig_t::vector_view_t view(vecs.data(), dim, N, /*on_host=*/true);
  const std::string path =
      std::string(::testing::TempDir()) + "jasper_read_rotation.bin";

  auto write = [&](bool record, bool prerotate, uint64_t seed, bool with_lsh) {
    fmt_ig_t ig;
    ig.n_vectors = N;
    ig.dim       = dim;
    ig.edges.assign(N, fmt_ig_t::fin_edge_list_t{});
    ig.counts.assign(N, 0);
    if (with_lsh) {
      ig.dir.has_lsh = true;
      ig.dir.edge_lshs.assign(N, edge_lsh_t{});
    }
    ig.dir.rotation.known     = record;
    ig.dir.rotation.prerotate = prerotate;
    ig.dir.rotation.seed      = seed;
    ig.save_directional_to_file(view, path);
  };

  // Rotated, with an LSH block in between — the plain config can't size that
  // block, which is why the record is located from EOF instead.
  write(true, true, 0x1234'5678'9ABCull, /*with_lsh=*/true);
  {
    auto info = read_rotation_info<plain_cfg>(path, dim);
    EXPECT_TRUE(info.known);
    EXPECT_TRUE(info.prerotate);
    EXPECT_EQ(info.seed, 0x1234'5678'9ABCull);
    // The directional config must of course agree.
    auto same = read_rotation_info<fmt_cfg>(path, dim);
    EXPECT_EQ(same.known, info.known);
    EXPECT_EQ(same.prerotate, info.prerotate);
    EXPECT_EQ(same.seed, info.seed);
  }

  // Recorded as NOT rotated: known, so a reader must NOT rotate queries.
  write(true, false, 0, /*with_lsh=*/true);
  {
    auto info = read_rotation_info<plain_cfg>(path, dim);
    EXPECT_TRUE(info.known);
    EXPECT_FALSE(info.prerotate);
  }

  // Trailer present but no rotation block → says nothing.
  write(false, false, 0, /*with_lsh=*/true);
  EXPECT_FALSE(read_rotation_info<plain_cfg>(path, dim).known);

  // No trailer at all (a plain index) → says nothing, and must not throw:
  // this is the common case for every index built before the block existed.
  write(false, false, 0, /*with_lsh=*/false);
  EXPECT_FALSE(read_rotation_info<plain_cfg>(path, dim).known);

  std::remove(path.c_str());
}

// A build that requested no artifacts writes a plain base section, so an old
// loader (and load_graph_from_file) still sees exactly the same file.
TEST(DiskannDirectionalSave, NoArtifactsMeansNoTrailer) {
  constexpr uint32_t dim = 8;
  constexpr uint32_t N   = 10;

  std::vector<__half> vecs(static_cast<size_t>(N) * dim, __float2half(1.0f));
  fmt_ig_t::vector_view_t view(vecs.data(), dim, N, /*on_host=*/true);

  fmt_ig_t ig;
  ig.n_vectors = N;
  ig.dim       = dim;
  ig.edges.assign(N, fmt_ig_t::fin_edge_list_t{});
  ig.counts.assign(N, 0);

  const std::string path =
      std::string(::testing::TempDir()) + "jasper_diskann_no_trailer.bin";
  ig.save_directional_to_file(view, path);

  const uint64_t bpn = sizeof(__half) * dim + sizeof(uint8_t)
                     + sizeof(fmt_ig_t::fin_edge_list_t);
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(f.is_open());
  EXPECT_EQ(static_cast<uint64_t>(f.tellg()), 4 * sizeof(uint64_t) + uint64_t{N} * bpn);
  f.close();
  std::remove(path.c_str());
}

}  // namespace jasper::test
