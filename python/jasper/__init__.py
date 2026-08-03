"""Jasper — GPU graph-based nearest neighbor search."""

from pathlib import Path
from enum import Enum

import torch
import tvm_ffi
import struct
import numpy as np
import re

# ── Load the compiled shared library ────────────────────────────
_lib_path = str(Path(__file__).parent / "lib" / "libjasper_ffi.so")
_mod: tvm_ffi.Module = tvm_ffi.load_module(_lib_path)


# ── Config enums ────────────────────────────────────────────────
class DistanceFunc(str, Enum):
    L2 = "l2"
    INNER_PRODUCT = "ip"


class DataType(str, Enum):
    FLOAT16 = "f16"


# ── Config registry (must match JASPER_FOR_EACH_CONFIG in C++) ──
# Index storage is __half; callers must pass torch.float16 tensors.
_CONFIGS: dict[tuple[DataType, int, DistanceFunc], str] = {
    (DataType.FLOAT16, 32,  DistanceFunc.L2):             "f16_r32_l2",
    (DataType.FLOAT16, 64,  DistanceFunc.L2):             "f16_r64_l2",
    (DataType.FLOAT16, 32,  DistanceFunc.INNER_PRODUCT):  "f16_r32_ip",
    (DataType.FLOAT16, 64,  DistanceFunc.INNER_PRODUCT):  "f16_r64_ip",
}

_DTYPE_MAP = {
    DataType.FLOAT16: torch.float16,
}


def _resolve_config(
    data_type: DataType, n_neighbors: int, distance: DistanceFunc
) -> str:
    key = (data_type, n_neighbors, distance)
    if key not in _CONFIGS:
        supported = "\n  ".join(
            f"({d.value}, R={r}, {f.value})" for (d, r, f) in _CONFIGS
        )
        raise ValueError(
            f"Unsupported config: data_type={data_type.value}, "
            f"n_neighbors={n_neighbors}, distance={distance.value}\n"
            f"Supported configs:\n  {supported}"
        )
    return _CONFIGS[key]


# ── Directional (LSH + PQ) config registry ───────────────────────
# Mirrors JASPER_FOR_EACH_DIRECTIONAL_CONFIG in ffi/jasper_ffi_common.cuh. Unlike the
# plain registry, the same (data_type, n_neighbors, distance, k_ranks) can map
# to different compiled configs depending on dim — PACKED_T (7-bit vs 15-bit
# packed coordinates) is chosen by dim bucket.
_DIRECTIONAL_CONFIGS: dict[tuple[DataType, int, DistanceFunc, int, bool], str] = {
    # key: (data_type, n_neighbors, distance, k_ranks, dim <= 128)
    (DataType.FLOAT16, 32, DistanceFunc.L2,            4,  True):  "f16_r32_l2_k4_d128",
    (DataType.FLOAT16, 32, DistanceFunc.L2,            16, True):  "f16_r32_l2_k16_d128",
    (DataType.FLOAT16, 32, DistanceFunc.L2,            4,  False): "f16_r32_l2_k4_d32678",
    (DataType.FLOAT16, 32, DistanceFunc.L2,            16, False): "f16_r32_l2_k16_d32678",
    (DataType.FLOAT16, 64, DistanceFunc.L2,            4,  True):  "f16_r64_l2_k4_d128",
    (DataType.FLOAT16, 64, DistanceFunc.L2,            16, True):  "f16_r64_l2_k16_d128",
    (DataType.FLOAT16, 64, DistanceFunc.L2,            4,  False): "f16_r64_l2_k4_d32678",
    (DataType.FLOAT16, 64, DistanceFunc.L2,            16, False): "f16_r64_l2_k16_d32678",
    (DataType.FLOAT16, 32, DistanceFunc.INNER_PRODUCT, 4,  True):  "f16_r32_ip_k4_d128",
    (DataType.FLOAT16, 32, DistanceFunc.INNER_PRODUCT, 16, True):  "f16_r32_ip_k16_d128",
    (DataType.FLOAT16, 32, DistanceFunc.INNER_PRODUCT, 4,  False): "f16_r32_ip_k4_d32678",
    (DataType.FLOAT16, 32, DistanceFunc.INNER_PRODUCT, 16, False): "f16_r32_ip_k16_d32678",
    (DataType.FLOAT16, 64, DistanceFunc.INNER_PRODUCT, 4,  True):  "f16_r64_ip_k4_d128",
    (DataType.FLOAT16, 64, DistanceFunc.INNER_PRODUCT, 16, True):  "f16_r64_ip_k16_d128",
    (DataType.FLOAT16, 64, DistanceFunc.INNER_PRODUCT, 4,  False): "f16_r64_ip_k4_d32678",
    (DataType.FLOAT16, 64, DistanceFunc.INNER_PRODUCT, 16, False): "f16_r64_ip_k16_d32678",
}


def _resolve_directional_config(
    data_type: DataType, n_neighbors: int, distance: DistanceFunc,
    k_ranks: int, dim: int,
) -> str:
    key = (data_type, n_neighbors, distance, k_ranks, dim <= 128)
    if key not in _DIRECTIONAL_CONFIGS:
        supported = "\n  ".join(
            f"({d.value}, R={r}, {f.value}, k_ranks={kr}, dim<=128={le128})"
            for (d, r, f, kr, le128) in _DIRECTIONAL_CONFIGS
        )
        raise ValueError(
            f"Unsupported directional config: data_type={data_type.value}, "
            f"n_neighbors={n_neighbors}, distance={distance.value}, "
            f"k_ranks={k_ranks}, dim={dim}\n"
            f"Supported configs:\n  {supported}"
        )
    return _DIRECTIONAL_CONFIGS[key]


# ── FFI function cache ──────────────────────────────────────────
_fn_cache: dict[str, tuple] = {}

_free_graph_fn = _mod.jasper_free_graph
_get_n_vectors_fn = _mod.jasper_get_n_vectors
_get_n_tombstoned_fn = _mod.jasper_get_n_tombstoned
_get_n_live_fn = _mod.jasper_get_n_live
_reserve_ids_fn = _mod.jasper_reserve_ids
_get_dim_fn = _mod.jasper_get_dim


def _get_fns(config_id: str):
    if config_id not in _fn_cache:
        _fn_cache[config_id] = (
            getattr(_mod, f"jasper_load_{config_id}"),
            getattr(_mod, f"jasper_construct_{config_id}"),
            getattr(_mod, f"jasper_search_{config_id}"),
            getattr(_mod, f"jasper_save_{config_id}"),
            getattr(_mod, f"jasper_get_vector_{config_id}"),
            getattr(_mod, f"jasper_mark_deleted_{config_id}"),
            getattr(_mod, f"jasper_consolidate_{config_id}"),
            getattr(_mod, f"jasper_compact_{config_id}"),
            getattr(_mod, f"jasper_append_{config_id}"),
        )
    return _fn_cache[config_id]


_directional_fn_cache: dict[str, tuple] = {}


def _get_directional_fns(config_id: str):
    if config_id not in _directional_fn_cache:
        _directional_fn_cache[config_id] = (
            getattr(_mod, f"jasper_construct_directional_{config_id}"),
            getattr(_mod, f"jasper_build_lsh_{config_id}"),
            getattr(_mod, f"jasper_build_pq_{config_id}"),
            getattr(_mod, f"jasper_directional_search_{config_id}"),
            getattr(_mod, f"jasper_pq_search_{config_id}"),
            getattr(_mod, f"jasper_save_directional_{config_id}"),
            getattr(_mod, f"jasper_load_directional_{config_id}"),
            getattr(_mod, f"jasper_get_vector_directional_{config_id}"),
            getattr(_mod, f"jasper_get_directional_flags_{config_id}"),
        )
    return _directional_fn_cache[config_id]

# ── Parse storage util ──────────────────────────────────────────
STORAGE_SIZE_RE = re.compile(r'^\s*([\d.]+)\s*([KMGTP]?i?B)\s*$', re.IGNORECASE)

STORAGE_UNITS = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}

def parse_storage_size(s: str) -> int:
    m = STORAGE_SIZE_RE.match(s)
    if not m:
        raise ValueError(f"Invalid storage size: {s}")
    value, unit = m.groups()
    return int(float(value) * STORAGE_UNITS[unit.upper()])

# ── Graph class ─────────────────────────────────────────────────
class Graph:
    """
    A GPU-resident graph index for nearest neighbor search.

    Usage:
        # Load from file
        g = jasper.Graph.load("sift1m.graph", dim=128, n_neighbors=32)

        # Build from vectors
        vectors = torch.randn(100000, 128, device="cuda")
        g = jasper.Graph.build(vectors, n_neighbors=32)

        # Search
        indices, distances = g.search(queries, k=10)
    """

    def __init__(
        self,
        handle: int,
        config_id: str,
        data_type: DataType,
        distance: DistanceFunc,
        n_neighbors: int,
        dim: int,
        *,
        is_directional: bool = False,
        k_ranks: int | None = None,
        has_lsh: bool = False,
        has_pq: bool = False,
        prerotate: bool = False,
        prerotate_seed: int = 42,
    ):
        self._handle = handle
        self._config_id = config_id
        self._data_type = data_type
        self._distance = distance
        self._n_neighbors = n_neighbors
        self._dim = dim
        self._torch_dtype = _DTYPE_MAP[data_type]
        self._is_directional = is_directional
        self._k_ranks = k_ranks
        self._has_lsh = has_lsh
        self._has_pq = has_pq
        self._prerotate = prerotate
        self._prerotate_seed = prerotate_seed

        if is_directional:
            (self._construct_fn, self._build_lsh_fn, self._build_pq_fn,
             self._directional_search_fn, self._pq_search_fn,
             self._save_fn, self._load_fn, self._get_vector_fn,
             self._get_flags_fn) = _get_directional_fns(config_id)
            # Deletion/append are only exported per-config for plain graphs —
            # directional configs don't have mark_deleted/consolidate/compact/
            # append FFI bindings (see ffi/jasper_ffi_common.cuh DEFINE_OPS vs
            # DEFINE_DIRECTIONAL_OPS).
            self._mark_deleted_fn = None
            self._consolidate_fn = None
            self._compact_fn = None
            self._append_fn = None
        else:
            (_, _, self._search_fn, self._save_fn, self._get_vector_fn,
             self._mark_deleted_fn, self._consolidate_fn,
             self._compact_fn, self._append_fn) = _get_fns(config_id)

    @classmethod
    def load(
        cls,
        path: str,
        dim: int,
        n_neighbors: int = 32,
        data_type: DataType | str = DataType.FLOAT16,
        distance: DistanceFunc | str = DistanceFunc.L2,
        on_host: bool = False,
        k_ranks: int | None = None,
        prerotate: bool = False,
        prerotate_seed: int = 42,
    ) -> "Graph":
        """
        Load a graph from a binary file into GPU memory.

        Args:
            path:         Path to the graph binary file.
            dim:          Dimensionality of the vectors.
            n_neighbors:  Max neighbors per node (must match file).
            data_type:    Vector data type: "f16".
            distance:     Distance function: "l2" or "ip".
            on_host:      Load the graph on host memory.
            k_ranks:        Pass the k_ranks used at build() time to load a
                           directional graph. Its on-disk trailer (see
                           Graph.build) is read automatically: whichever of
                           directional_search()/pq_search() have their
                           artifacts present become usable. Leave None to
                           load a plain graph.
            prerotate:     Must match what build() used for this graph — not
                           stored in the file. Only relevant when k_ranks is
                           given.
                           FOOTGUN: build()'s default is
                           `prerotate = build_lsh or build_pq` (usually True),
                           but this default is False. If you built with
                           prerotate on (the common case) and load() without
                           passing prerotate=True, queries silently skip
                           rotation against a rotated index — results are
                           wrong with no error. Always pass the same
                           prerotate/prerotate_seed used at build() time.
                           (Not stored in the trailer yet — see
                           save_directional_graph_to_file/
                           load_directional_graph_from_file in graph.cuh.)
            prerotate_seed: Must match what build() used for this graph.
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        if k_ranks is None:
            config_id = _resolve_config(data_type, n_neighbors, distance)
            load_fn = _get_fns(config_id)[0]
            handle = load_fn(path, dim, on_host)
            return cls(handle, config_id, data_type, distance, n_neighbors, dim)

        config_id = _resolve_directional_config(
            data_type, n_neighbors, distance, k_ranks, dim
        )
        (_, _, _, _, _, _, load_fn, _, get_flags_fn) = _get_directional_fns(config_id)
        handle = load_fn(path, dim, on_host, prerotate, prerotate_seed)
        flags = get_flags_fn(handle)

        return cls(
            handle, config_id, data_type, distance, n_neighbors, dim,
            is_directional=True, k_ranks=k_ranks,
            has_lsh=bool(flags & 1), has_pq=bool(flags & 2),
            prerotate=prerotate, prerotate_seed=prerotate_seed,
        )
    
    def save(self, path: str) -> None:
        """
        Save this graph to a binary file.

        The file can later be reloaded with ``Graph.load()``, using the
        same ``dim``, ``n_neighbors``, ``data_type``, and ``distance``
        that were used when the graph was built or originally loaded.

        Args:
            path: Destination file path.
        """
        self._check_alive()
        self._save_fn(self._handle, path)

    @classmethod
    def build(
        cls,
        vectors: torch.Tensor,
        n_neighbors: int = 32,
        distance: DistanceFunc | str = DistanceFunc.L2,
        alpha: float = 1.2,
        workspace_budget: str = "10GB",
        on_host: bool = False,
        build_lsh: bool = False,
        build_pq: bool = False,
        k_ranks: int = 4,
        prerotate: bool | None = None,
        prerotate_seed: int = 42,
        lsh_samples: int = 32768,
        lsh_seed: int = 42,
        pq_train: int = 40000,
        pq_kmeans_iter: int = 12,
        pq_seed: int = 123,
    ) -> "Graph":
        """
        Construct a graph index from vectors on GPU.

        Args:
            vectors:          CUDA tensor of shape [n_vectors, dim].
            n_neighbors:      Max neighbors per node (R).
            distance:         Distance function: "l2" or "ip".
            alpha:            Pruning factor (1.0 = strict, >1.0 = long hops).
            workspace_budget: GPU memory budget (default to 10GB).
            on_host:          Construct the graph on host memory.
            build_lsh:        Also populate cross-polytope LSH edges, enabling
                             directional_search().
            build_pq:         Also populate Product-Quantization edges + exact
                             vector norms, enabling pq_search().
            k_ranks:          LSH rank count / PQ subquantizer count (4 or 16).
                             Only used when build_lsh or build_pq is True.
            prerotate:        Rotate vectors (and, transparently, queries at
                             search time) by a random orthogonal matrix.
                             Defaults to True iff build_lsh or build_pq is
                             set, since LSH/PQ estimator quality relies on
                             it — pass False explicitly to disable.
                             NOTE: this choice (and prerotate_seed) is NOT
                             persisted by save()/load() — see the
                             ⚠ FOOTGUN note on Graph.load's prerotate arg.
            prerotate_seed:   Seed for the rotation matrix.
            lsh_samples:      Edge samples used to calibrate LSH globals.
            lsh_seed:         Seed for LSH global sampling.
            pq_train:         Residual samples used to train PQ codebooks.
            pq_kmeans_iter:   K-means iterations for PQ codebook training.
            pq_seed:          Seed for PQ codebook training.

        Returns:
            A Graph ready for search() (plain graphs), or additionally for
            directional_search()/pq_search() when build_lsh/build_pq are set.
        """
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        workspace_budget_bytes = parse_storage_size(workspace_budget)

        # if not vectors.is_cuda:
        #     raise ValueError("vectors must be a CUDA tensor")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got {vectors.ndim}D")

        vectors = vectors.contiguous()
        dim = vectors.size(1)

        # Infer data type from tensor dtype
        if vectors.dtype == torch.float16:
            data_type = DataType.FLOAT16
        else:
            raise ValueError(
                f"Unsupported tensor dtype: {vectors.dtype}. "
                f"Use torch.float16."
            )

        if not (build_lsh or build_pq):
            config_id = _resolve_config(data_type, n_neighbors, distance)
            construct_fn = _get_fns(config_id)[1]
            handle = construct_fn(vectors, dim, alpha, workspace_budget_bytes, on_host)
            return cls(handle, config_id, data_type, distance, n_neighbors, dim)

        config_id = _resolve_directional_config(
            data_type, n_neighbors, distance, k_ranks, dim
        )
        resolved_prerotate = prerotate if prerotate is not None else True

        (construct_fn, build_lsh_fn, build_pq_fn, _, _, _, _, _, _) = \
            _get_directional_fns(config_id)

        handle = construct_fn(
            vectors, dim, alpha, workspace_budget_bytes, on_host,
            resolved_prerotate, prerotate_seed,
        )

        if build_lsh:
            build_lsh_fn(handle, lsh_samples, lsh_seed)
        if build_pq:
            build_pq_fn(handle, pq_train, pq_kmeans_iter, pq_seed)

        return cls(
            handle, config_id, data_type, distance, n_neighbors, dim,
            is_directional=True, k_ranks=k_ranks,
            has_lsh=build_lsh, has_pq=build_pq,
            prerotate=resolved_prerotate, prerotate_seed=prerotate_seed,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_vectors(self) -> int:
        self._check_alive()
        return _get_n_vectors_fn(self._handle)

    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def distance(self) -> DistanceFunc:
        return self._distance

    def search(
        self,
        queries: torch.Tensor,
        k: int = 10,
        beam_width: int = 64,
        print_throughput: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search on this graph.

        Args:
            queries:    CUDA tensor of shape [n_queries, dim].
            k:          Number of nearest neighbors to return.
            beam_width: Search beam width.
        Returns:
            indices:   int32  tensor [n_queries, k] of **stable ids**
            distances: float32 tensor [n_queries, k]
        """
        self._check_alive()
        if self._is_directional:
            raise RuntimeError(
                "search() is not available on a directional graph "
                "(built with build_lsh=True/build_pq=True) — use "
                "directional_search() or pq_search() instead"
            )

        if not queries.is_cuda:
            raise ValueError("queries must be a CUDA tensor")
        if queries.dtype != self._torch_dtype:
            raise ValueError(
                f"queries must be {self._torch_dtype}, got {queries.dtype}"
            )
        if queries.ndim != 2:
            raise ValueError(f"queries must be 2D, got {queries.ndim}D")
        if queries.size(1) != self._dim:
            raise ValueError(
                f"query dim ({queries.size(1)}) != graph dim ({self._dim})"
            )

        queries = queries.contiguous()
        n_queries = queries.size(0)

        out_indices = torch.empty(
            n_queries, k, dtype=torch.int32, device=queries.device
        )
        out_distances = torch.empty(
            n_queries, k, dtype=torch.float32, device=queries.device
        )

        # limit is automaticall set to 2x the beam_width
        limit = beam_width * 2

        self._search_fn(
            self._handle, queries, out_indices, out_distances,
            k, beam_width, limit, print_throughput
        )

        return out_indices, out_distances

    def _directional_query_check(self, queries: torch.Tensor) -> torch.Tensor:
        if not queries.is_cuda:
            raise ValueError("queries must be a CUDA tensor")
        if queries.dtype != self._torch_dtype:
            raise ValueError(
                f"queries must be {self._torch_dtype}, got {queries.dtype}"
            )
        if queries.ndim != 2:
            raise ValueError(f"queries must be 2D, got {queries.ndim}D")
        if queries.size(1) != self._dim:
            raise ValueError(
                f"query dim ({queries.size(1)}) != graph dim ({self._dim})"
            )
        return queries.contiguous()

    def directional_search(
        self,
        queries: torch.Tensor,
        k: int = 10,
        beam_width: int = 64,
        limit: int | None = None,
        print_throughput: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search scored by the cross-polytope LSH estimator.

        Requires this graph to have been built with build_lsh=True (or
        loaded with LSH artifacts present — see Graph.load's k_ranks arg).
        Query rotation (if this graph was prerotated) happens transparently.

        Args:
            queries:    CUDA tensor of shape [n_queries, dim].
            k:          Number of nearest neighbors to return.
            beam_width: Search beam width.
            limit:      Defaults to 2x beam_width.
        Returns:
            indices:   int32  tensor [n_queries, k]
            distances: float32 tensor [n_queries, k]
        """
        self._check_alive()
        if not self._is_directional or not self._has_lsh:
            raise RuntimeError(
                "directional_search() requires a graph built with "
                "build_lsh=True (or loaded with LSH artifacts present)"
            )

        queries = self._directional_query_check(queries)
        n_queries = queries.size(0)

        out_indices = torch.empty(
            n_queries, k, dtype=torch.int32, device=queries.device
        )
        out_distances = torch.empty(
            n_queries, k, dtype=torch.float32, device=queries.device
        )

        resolved_limit = limit if limit is not None else beam_width * 2
        self._directional_search_fn(
            self._handle, queries, out_indices, out_distances,
            k, beam_width, resolved_limit, print_throughput,
        )

        return out_indices, out_distances

    def pq_search(
        self,
        queries: torch.Tensor,
        k: int = 10,
        beam_width: int = 64,
        limit: int | None = None,
        print_throughput: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search scored by the Product-Quantization ADC estimator.

        Requires this graph to have been built with build_pq=True (or
        loaded with PQ artifacts present — see Graph.load's k_ranks arg).
        Query rotation (if this graph was prerotated) happens transparently.

        Args:
            queries:    CUDA tensor of shape [n_queries, dim].
            k:          Number of nearest neighbors to return.
            beam_width: Search beam width.
            limit:      Defaults to 2x beam_width.
        Returns:
            indices:   int32  tensor [n_queries, k]
            distances: float32 tensor [n_queries, k]
        """
        self._check_alive()
        if not self._is_directional or not self._has_pq:
            raise RuntimeError(
                "pq_search() requires a graph built with build_pq=True "
                "(or loaded with PQ artifacts present)"
            )

        queries = self._directional_query_check(queries)
        n_queries = queries.size(0)

        out_indices = torch.empty(
            n_queries, k, dtype=torch.int32, device=queries.device
        )
        out_distances = torch.empty(
            n_queries, k, dtype=torch.float32, device=queries.device
        )

        resolved_limit = limit if limit is not None else beam_width * 2
        self._pq_search_fn(
            self._handle, queries, out_indices, out_distances,
            k, beam_width, resolved_limit, print_throughput,
        )

        return out_indices, out_distances

    def get_vector(self, stable_id: int) -> torch.Tensor:
        """
        Return the vector with the given **stable id**.

        Stable ids are assigned in monotonic order as vectors are added and are
        unchanged by ``consolidate``/``compact`` — they are not bounded by
        ``n_vectors``. Raises if the id is not present (deleted or never assigned).

        Args:
            stable_id: The vector's stable id (as returned by ``search``).

        Returns:
            A 1-D CUDA tensor of shape [dim] with the graph's data type.
        """
        self._check_alive()
        if stable_id < 0:
            raise IndexError(f"stable_id must be non-negative, got {stable_id}")
        out = torch.empty(self._dim, dtype=self._torch_dtype, device="cuda")
        self._get_vector_fn(self._handle, stable_id, out)
        return out

    @property
    def n_tombstoned(self) -> int:
        """Number of soft-deleted vectors not yet consolidated away."""
        self._check_alive()
        return _get_n_tombstoned_fn(self._handle)

    @property
    def n_live(self) -> int:
        """Number of vectors still live (n_vectors - n_tombstoned)."""
        self._check_alive()
        return _get_n_live_fn(self._handle)

    def reserve_ids(self, count: int) -> torch.Tensor:
        """
        Reserve ``count`` fresh stable ids, returning them as an int32 CPU
        tensor of shape [count]. Advances the monotonic id counter; ids are
        never reused. This is the id half of a live append — the caller writes
        the corresponding vectors into the graph and registers each (id, slot).

        Returns:
            int32 tensor [count] of newly assigned stable ids.
        """
        self._check_alive()
        if count <= 0:
            return torch.empty(0, dtype=torch.int32)
        out = torch.empty(count, dtype=torch.int32)
        _reserve_ids_fn(self._handle, out, count)
        return out

    def append(self, vectors: torch.Tensor, alpha: float = 1.2) -> torch.Tensor:
        """
        Append a batch of vectors to the live graph, wiring their edges into the
        existing graph (beam-search + robust-prune, same as construction) and
        assigning each a fresh monotonic stable id.

        Args:
            vectors: CUDA tensor [n, dim] of the graph's data type (float16).
            alpha:   Robust-pruning factor for the new vectors' edges.

        Returns:
            int32 tensor [n] of the assigned stable ids, in input order.
        """
        self._check_alive()
        if not vectors.is_cuda:
            raise ValueError("vectors must be a CUDA tensor")
        if vectors.dtype != self._torch_dtype:
            raise ValueError(f"vectors must be {self._torch_dtype}, got {vectors.dtype}")
        if vectors.ndim != 2 or vectors.size(1) != self._dim:
            raise ValueError(f"vectors must be [n, {self._dim}]")
        vectors = vectors.contiguous()
        n = vectors.size(0)
        out_ids = torch.empty(n, dtype=torch.int32)
        self._append_fn(self._handle, vectors, float(alpha), out_ids)
        return out_ids

    def mark_deleted(self, ids: torch.Tensor) -> None:
        """
        Soft-delete a batch of vectors by id.

        Deleted vectors are immediately excluded from ``search`` results.
        Their graph edges are repaired lazily by ``consolidate`` and their
        slots reclaimed by ``compact``. Out-of-range ids are ignored.

        Args:
            ids: 1-D integer tensor of vector ids to delete.
        """
        self._check_alive()
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, got {ids.ndim}D")
        # FFI reads ids as a contiguous CPU int32 array.
        ids = ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
        if ids.numel() == 0:
            return
        self._mark_deleted_fn(self._handle, ids)

    def consolidate(self, alpha: float = 1.2) -> None:
        """
        Repair graph edges that route through deleted vertices and clear all
        tombstones. After this call ``n_tombstoned`` is 0; ids are unchanged.

        Args:
            alpha: Robust-pruning factor used when re-selecting edges.
        """
        self._check_alive()
        self._consolidate_fn(self._handle, float(alpha))

    def compact(self) -> None:
        """
        Reclaim space by compacting live vectors into internal slots
        ``[0, n_live)``. Consolidates first if there are pending deletions.

        Stable ids are preserved: a vector keeps the same id across ``compact``
        (only internal slots are renumbered, transparently via the id map).
        """
        self._check_alive()
        self._compact_fn(self._handle)

    def free(self):
        if self._handle is not None:
            _free_graph_fn(self._handle)
            self._handle = None

    def _check_alive(self):
        if self._handle is None:
            raise RuntimeError("Graph has been freed")

    def __del__(self):
        self.free()

    def __repr__(self) -> str:
        if self._handle is None:
            return "Graph(freed)"
        extra = ""
        if self._is_directional:
            extra = (
                f", k_ranks={self._k_ranks}, lsh={self._has_lsh}, "
                f"pq={self._has_pq}, prerotate={self._prerotate}"
            )
        return (
            f"Graph(n_vectors={self.n_vectors}, dim={self.dim}, "
            f"R={self._n_neighbors}, dtype={self._data_type.value}, "
            f"dist={self._distance.value}{extra})"
        )

# ── Utils ───────────────────────────────────────────────────────

def read_bin(path: str, dtype: str = "f32", max_vectors: int = 0) -> torch.Tensor:
    """Read a [n, dim] binary file (f32 or u8 on disk) and return a
    pinned torch.float16 tensor."""
    if dtype == "f32":
        np_dtype = np.float32
    elif dtype == "u8":
        np_dtype = np.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype!r}. Expected 'f32' or 'u8'.")

    with open(path, "rb") as f:
        n_vectors, dim = struct.unpack("II", f.read(8))
        if max_vectors > 0:
            n_vectors = min(n_vectors, max_vectors)
        nbytes = n_vectors * dim * np.dtype(np_dtype).itemsize
        data = np.frombuffer(f.read(nbytes), dtype=np_dtype).reshape(n_vectors, dim).copy()

    return torch.from_numpy(data).to(torch.float16).pin_memory()

def read_groundtruth(path: str, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read ground truth from a binary file.

    Format: [n_queries: uint32][gt_k: uint32][ids: n_queries * gt_k * uint32][distances: n_queries * gt_k * float32]

    Args:
        path: Path to the ground truth .bin file.
        k:    Number of neighbors to return (must be <= gt_k in file).

    Returns:
        indices:   int32 tensor [n_queries, k]
        distances: float32 tensor [n_queries, k]
    """
    with open(path, "rb") as f:
        n_queries, gt_k = struct.unpack("II", f.read(8))

        if gt_k < k:
            raise ValueError(
                f"Requested k={k} but ground truth only has k={gt_k}"
            )

        ids = np.frombuffer(f.read(n_queries * gt_k * 4), dtype=np.uint32).reshape(n_queries, gt_k)
        distances = np.frombuffer(f.read(n_queries * gt_k * 4), dtype=np.float32).reshape(n_queries, gt_k)

    return (
        torch.from_numpy(ids[:, :k].copy()).to(dtype=torch.int32),
        torch.from_numpy(distances[:, :k].copy()),
    )

def get_recall(gt, result_indices, k, n_queries):
    count = sum(
        len(set(result_indices[i].tolist()) & set(gt[i].tolist()))
        for i in range(n_queries)
    )
    recall = count / (n_queries * k)
    print(f"Recall@{k}: {recall:.3f} ({count}/{(n_queries * k)})")
    return recall

def generate_groundtruth(
    vectors: torch.Tensor,
    queries: torch.Tensor,
    k: int = 100,
    distance: str = "l2",
    query_batch_size: int = 1024,
    vector_batch_size: int = 100_000,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Brute-force exact k-NN on GPU, streaming both vectors and queries
    in batches to stay within device memory.

    Args:
        vectors:           [n, dim] CPU tensor (float32 or float16) — the database.
        queries:           [nq, dim] CPU tensor (float32 or float16) — the queries.
        k:                 Number of nearest neighbors.
        distance:          "l2" or "ip" (inner product).
        query_batch_size:  Queries transferred to device per batch.
        vector_batch_size: Vectors transferred to device per batch.
        device:            Target device.

    Returns:
        indices:   int32   [nq, k]  (CPU)
        distances: float32 [nq, k]  (CPU)
    """
    assert vectors.dtype in (torch.float32, torch.float16), \
        f"vectors must be float32 or float16, got {vectors.dtype}"
    assert queries.dtype in (torch.float32, torch.float16), \
        f"queries must be float32 or float16, got {queries.dtype}"
    assert vectors.size(1) == queries.size(1)

    # Promote to float32 for L2 accumulation to avoid fp16 overflow;
    # for IP, fp16 matmul is fine and we only upcast the final distances.
    use_fp16 = vectors.dtype == torch.float16 and queries.dtype == torch.float16
    compute_dtype = torch.float16 if (use_fp16 and distance == "ip") else torch.float32

    n, dim = vectors.shape
    nq = queries.size(0)

    # Final results live on CPU in float32
    final_ids = torch.empty(nq, k, dtype=torch.int32)
    final_dists = torch.empty(nq, k, dtype=torch.float32)

    n_query_batches = (nq + query_batch_size - 1) // query_batch_size
    n_vector_batches = (n + vector_batch_size - 1) // vector_batch_size

    for qi, q_start in enumerate(range(0, nq, query_batch_size)):
        q_end = min(q_start + query_batch_size, nq)
        q_batch = queries[q_start:q_end].to(device=device, dtype=compute_dtype)  # [qbs, dim]
        qbs = q_batch.size(0)

        if distance == "l2":
            q_sq = (q_batch * q_batch).sum(dim=1, keepdim=True)  # [qbs, 1]

        # Accumulate top-k across vector batches on device (always float32)
        top_dists = torch.full((qbs, k), float("inf"), dtype=torch.float32, device=device)
        top_ids = torch.zeros((qbs, k), dtype=torch.int64, device=device)

        for vi, v_start in enumerate(range(0, n, vector_batch_size)):
            print(
                f"\r[groundtruth] query batch {qi+1}/{n_query_batches}, "
                f"vector batch {vi+1}/{n_vector_batches}",
                end="", flush=True,
            )
            v_end = min(v_start + vector_batch_size, n)
            v_batch = vectors[v_start:v_end].to(device=device, dtype=compute_dtype)  # [vbs, dim]

            if distance == "l2":
                v_sq = (v_batch * v_batch).sum(dim=1)  # [vbs]
                dists = q_sq + v_sq.unsqueeze(0) - 2.0 * (q_batch @ v_batch.T)
                dists.clamp_(min=0.0)
            elif distance == "ip":
                dists = -(q_batch @ v_batch.T)  # negate so smaller = better
            else:
                raise ValueError(f"Unknown distance: {distance}")

            # Upcast to float32 before merging into top-k accumulator
            dists = dists.float()

            # Offset local indices to global vector indices
            local_ids = torch.arange(v_start, v_end, device=device).unsqueeze(0).expand(qbs, -1)

            # Merge with running top-k: concat then re-select top-k
            merged_dists = torch.cat([top_dists, dists], dim=1)     # [qbs, k + vbs]
            merged_ids = torch.cat([top_ids, local_ids], dim=1)     # [qbs, k + vbs]

            top_dists, sel = merged_dists.topk(k, dim=1, largest=False)
            top_ids = merged_ids.gather(1, sel)

            del v_batch, dists, local_ids, merged_dists, merged_ids, sel

        final_ids[q_start:q_end] = top_ids.to(torch.int32).cpu()
        final_dists[q_start:q_end] = top_dists.cpu()

        del q_batch, top_dists, top_ids

    print(f"\r[groundtruth] done — {nq} queries, {n} vectors, k={k}" + " " * 20)
    return final_ids, final_dists


def save_groundtruth(
    path: str,
    indices: torch.Tensor,
    distances: torch.Tensor,
):
    """
    Write ground truth to the binary format expected by `jasper.read_groundtruth`.

    Format: [n_queries: u32][k: u32][ids: u32 * nq * k][dists: f32 * nq * k]
    """
    ids_np = indices.cpu().numpy().astype(np.uint32)
    dists_np = distances.cpu().numpy().astype(np.float32)
    nq, k = ids_np.shape

    with open(path, "wb") as f:
        f.write(struct.pack("II", nq, k))
        f.write(ids_np.tobytes())
        f.write(dists_np.tobytes())

# ── Usage ───────────────────────────────────────────────────────
#
# import jasper
#
# # Load from file
# g = jasper.Graph.load("sift1m.graph", dim=128, n_neighbors=32)
# indices, distances = g.search(queries, k=10)
#
# # Build from vectors
# vectors = torch.randn(1_000_000, 128, device="cuda")
# g = jasper.Graph.build(vectors, n_neighbors=64, alpha=1.2)
# indices, distances = g.search(queries, k=10)
#
# # Directional (LSH / PQ) build + search
# g = jasper.Graph.build(vectors, n_neighbors=64, build_lsh=True, build_pq=True)
# indices, distances = g.directional_search(queries, k=10)
# indices, distances = g.pq_search(queries, k=10)
# g.save("index.graph")  # persists LSH/PQ artifacts too
# g2 = jasper.Graph.load("index.graph", dim=128, n_neighbors=64, k_ranks=4)