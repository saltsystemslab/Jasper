# Directional Beam Search

## Compilation

Compile library
```bash
# Install tvm_ffi
pip install apache-tvm-ffi

# Build the FFI shared library
# CMAKE_CUDA_ARCHITECTURES defaults to 120; override for your target GPU
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON -DCMAKE_CUDA_ARCHITECTURES="90;120"
cmake --build build -j
cmake --install build
pip install -e python/
```

## Python Usage

All Python APIs can be found under the `example/` folder, where we provides python scripts to construct and query vector index in jasper. See `example/graph.py` for building and searching in one pass, `example/query.py` for loading a saved index and running queries, and `example/groundtruth.py` for generating exact k-NN ground truth to measure recall.

### Directional search (LSH / PQ estimators)

A **directional graph** stores extra per-edge estimator artifacts alongside
the plain graph, enabling beam search scored by a cheaper proxy distance
instead of the exact one. Build with `build_lsh=True` for a cross-polytope
LSH estimator (`directional_search()`), or `build_pq=True` for a
Product-Quantization ADC estimator (`pq_search()`). Both can also be
enabled together on the same graph and queried independently.

#### LSH estimator

```python
g = jasper.Graph.build(
    vectors,             # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,       # int    — max neighbors per node (R)
    distance="l2",         # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,               # float  — pruning factor (1.0=strict, >1.0=longer hops)
    build_lsh=True,           # bool — populate LSH edges, enables directional_search()
    k_ranks=4,                  # int — LSH rank count (4 or 16)
    prerotate=None,               # bool | None — rotate vectors/queries by a random
                                   #   orthogonal matrix; defaults to True since it's
                                   #   needed for LSH estimator quality
    lsh_samples=32768,             # int — edge samples used to calibrate LSH globals
)
# g: jasper.Graph, is_directional=True
# NOTE: g.search() raises on a directional graph — use directional_search() instead.

indices, distances = g.directional_search(
    queries,              # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,                  # int    — number of neighbors to return
    beam_width=64,          # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
# Requires build_lsh=True at build() time (or LSH artifacts present at load()).
```

#### PQ estimator

```python
g = jasper.Graph.build(
    vectors,             # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,       # int    — max neighbors per node (R)
    distance="l2",         # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,               # float  — pruning factor (1.0=strict, >1.0=longer hops)
    build_pq=True,            # bool — populate PQ edges + exact norms, enables pq_search()
    k_ranks=4,                  # int — PQ subquantizer count (4 or 16)
    prerotate=None,               # bool | None — rotate vectors/queries by a random
                                   #   orthogonal matrix; defaults to True since it's
                                   #   needed for PQ estimator quality
    pq_train=40000,                # int — residual samples used to train PQ codebooks
    pq_kmeans_iter=12,               # int — k-means iterations for PQ codebook training
)
# g: jasper.Graph, is_directional=True
# NOTE: g.search() raises on a directional graph — use pq_search() instead.

indices, distances = g.pq_search(
    queries,              # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,                  # int    — number of neighbors to return
    beam_width=64,          # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
# Requires build_pq=True at build() time (or PQ artifacts present at load()).
```

### Save a graph

```python
g.save(
    "sift1m.graph",  # str — destination file path
)
# returns: None
```

## Slurm benchmark examples

Slurm scripts for the benchmarks in the paper are stored in `sbatch/`. Every
script builds the binaries it needs with CMake before running, so submit it
from a checkout of this repo (the long-running sweeps assume `REPO=$HOME/jasper`).

Common to the long-running sweeps (the four `bench_deep_*`/`bench_locale*`
scripts and `distortion_sweep`):

- They pin `--nodelist=d4067` and run for up to 24 h. Every tunable is an
  environment variable, documented in the script's header comment.
- `DRY_RUN=1 ./sbatch/<script>.sbatch` prints the plan (shard count, per-cell
  host/device memory, whether each cell runs on the device or `--on_host`) and
  exits without building anything. Run it before submitting a large job.
- Indexes are built in node-local `/tmp`, which is wiped when the job ends, so
  build and query happen in the same job. Logs and results go to persistent
  storage under the repo.

### Edge-estimator experiments (CPH/LSH, PQ, RaBitQ)

| Script | What it measures |
|---|---|
| `bench_deep_ksweep.sbatch` | Sweeps code size `r` (`--k_ranks` = 4, 8, 16) for all three estimators on one plain DEEP index, so every cell searches the same graph. Reports QPS and recall@k with `run_query`. `SCALE=10M/100M/200M/1B` picks the dataset size and `R_INT`/`R_FIN` the degree (e.g. `SCALE=1B R_INT=32 R_FIN=32`). With `RUN_POPS=1` it also records nodes explored per query with `distortion_test --pops`. |
| `bench_locale_ksweep.sbatch` | The same sweep on the LOCALE genomic embeddings: `DATASET=sra50` or `sra500`, degree fixed at 64, with an optional query-mutation axis (`MUTATIONS="0.00 0.05 0.10"`). Supports `RUN_POPS=1`. SRA500 must first be converted from its Hugging Face bundle (see `scripts/locale_bundle_to_fbin.py` below). |
| `distortion_sweep.sbatch` | Estimator accuracy microbenchmarks with `distortion_test` on device-resident datasets (BigANN 10M, Deep 10M, GIST, OpenAI, SRA50). For each dataset, estimator and `r` it writes recall vs nodes popped, plus estimated-vs-exact edge distances for (query, popped node) pairs taken from real searches and for random (node, query) pairs. Then it draws the distortion scatters and ranking plots. |

### Search termination (beam width vs Adaptive Beam Search)

| Script | What it measures |
|---|---|
| `bench_adaptive_termination.sbatch` | Pareto comparison of conventional beam search and directional search with CPH/LSH, RaBitQ and PQ codes, all on one plain index. Conventional search sweeps beam width (16 to 1983 in ~20% steps, plus each compiled tier's widest width). Each directional method also sweeps the Adaptive Beam Search stop rule (Al-Jazzazi et al. 2025): stop once the next candidate's estimate is at least (1+γ)× the k-th best estimate among the other discovered candidates, at 9 candidate-list widths. They also run fixed beam search with a re-rank (`RERANK_MULTS`, M = 1, 2 and 4 × k): the best M candidates left unexpanded get an exact distance, one vector read each, and compete for the top-k. Each (dataset, method, `r`, k) cell runs at recall 10@10 and 100@100, with the graph on the device and on the host. `run_query` measures QPS and `distortion_test --pops` measures nodes expanded and vectors read. `scripts/plot_pareto.py` then draws one Pareto curve per method in each panel, and `scripts/adaptive_summary.py` compares the two stop rules within each directional method. `WIDE_TIER=1` builds with `-DJASPER_WIDE_TIER=ON` (in `build_wide/`), which gives LSH and RaBitQ search a 4096-entry candidate list, so beam width + re-rank can reach 4031; `ONLY_WIDE=1` runs just those points. Uses the `L192` two-pass plain indexes under `INDEX_DIR`, building any that are missing. |

### Graph build quality

| Script | What it measures |
|---|---|
| `profile_search.sbatch` | Where a search's time goes and how its pops spread over the graph, to pick the next optimization. Builds `run_query` with `-DJASPER_PROFILE_CLOCKS=ON` (every 64th query's block counts cycles per kernel phase: pop, fetch + exact, top-k insert, expand, estimate, duplicate check, sort, dedup, clip) and runs conventional, LSH and RaBitQ search at each compiled width's top, on device and host. Then `distortion_test --visit_stats` reports, per width, the share of pops a GPU-resident cache of the 0.1–20% most popped nodes would serve (ranked on half the queries, scored on the other half) and the pops before each query reaches its first true neighbor. `scripts/profile_summary.py` writes the tables. |
| `bench_graph_build.sbatch` | Whether a better graph build helps. Builds each dataset's plain index under each `VARIANTS` entry (`L<n>` = the build search's beam width, `_2pass` = Vamana's two-pass build, otherwise `--one_pass`; `create_diskann_index` now defaults to `L192` two-pass, which needed 0.3–0.8× the nodes expanded of the old `L64` one-pass build at matched recall), then measures recall against nodes expanded per query with `distortion_test --pops` for conventional and directional search at 10@10 and 100@100. `L` is a build-time constant (`cmake -DJASPER_BUILD_L=<n>`), so each distinct `L` gets its own build directory. `scripts/compare_builds.py` summarizes. |

### Index construction at scale (DiskANN-style shard and merge)

| Script | What it measures |
|---|---|
| `bench_deep_scaling.sbatch` | A scaling ladder on DEEP from 10M to the full 1B. Each rung builds a sharded index with `create_diskann_index` (codes populated per shard) and queries it with conventional beam search and with every estimator that was built. The quantity of interest is how much recall the shard-and-merge build gives up as N grows. At 1B, host RAM is the binding constraint, so check `DRY_RUN=1` first. |
| `bench_locale.sbatch` | The same build-and-query benchmark on LOCALE SRA50 and SRA500, across the three query mutation rates (0%, 5%, 10%). Every (mutation, estimator, k) combination queries one index per dataset. |

### Helper scripts used by the sweeps (`scripts/`)

| Script | Purpose |
|---|---|
| `locale_bundle_to_fbin.py` | Converts the LOCALE Hugging Face release (503 base shards plus per-mutation queries and ground truth) into the `base.fbin` / `query_mut*.fbin` / `gt_mut*.ibin` layout the locale scripts read. Run it once before `DATASET=sra500`. |
| `tile_queries.py` | Repeats a small query set (and its ground truth) up to a target batch size so the GPU is saturated. Recall is unchanged. |
| `plot_pops.py` | Recall vs nodes popped per query, from `--pops` CSVs. |
| `plot_pareto.py` | Pareto curves (QPS on device and host, nodes expanded, vectors read, each against recall on a logit axis) per method from `bench_adaptive_termination`'s joined CSV, plus the frontier points as CSV and a matched-recall table across methods. |
| `compare_builds.py` | For `bench_graph_build`: per method and k, each build variant's recall ceiling and the nodes expanded to reach each target recall (interpolated), relative to the first variant, plus build time and average degree. |
| `adaptive_summary.py` | Joins `run_query` throughput with `--pops` rows for `bench_adaptive_termination`. For each rule it gives the best QPS and the fewest pops at each target recall. |
| `plot_distortion.py` | Estimated-vs-true distance density grids, from edge-score CSVs. |
| `neighborhood_ranking.py` | Within-neighborhood ranking accuracy (recall@m, Spearman) vs bytes per edge, from edge-score CSVs. |
| `export_distortion_pgf.py` | Exports a distortion grid as pgfplots CSVs plus a ready-to-`\input` `.tex` figure. |
