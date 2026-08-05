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
    (DataType.FLOAT16, 8,   DistanceFunc.L2):             "f16_r8_l2",
    (DataType.FLOAT16, 16,  DistanceFunc.L2):             "f16_r16_l2",
    (DataType.FLOAT16, 32,  DistanceFunc.L2):             "f16_r32_l2",
    (DataType.FLOAT16, 64,  DistanceFunc.L2):             "f16_r64_l2",
    (DataType.FLOAT16, 8,   DistanceFunc.INNER_PRODUCT):  "f16_r8_ip",
    (DataType.FLOAT16, 16,  DistanceFunc.INNER_PRODUCT):  "f16_r16_ip",
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
    ):
        self._handle = handle
        self._config_id = config_id
        self._data_type = data_type
        self._distance = distance
        self._n_neighbors = n_neighbors
        self._dim = dim
        self._torch_dtype = _DTYPE_MAP[data_type]
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
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        config_id = _resolve_config(data_type, n_neighbors, distance)
        load_fn = _get_fns(config_id)[0]
        handle = load_fn(path, dim, on_host)

        return cls(handle, config_id, data_type, distance, n_neighbors, dim)
    
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
        on_host: bool = False
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

        Returns:
            A Graph ready for search.
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

        config_id = _resolve_config(data_type, n_neighbors, distance)
        construct_fn = _get_fns(config_id)[1]
        handle = construct_fn(vectors, dim, alpha, workspace_budget_bytes, on_host)

        return cls(handle, config_id, data_type, distance, n_neighbors, dim)

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
        return (
            f"Graph(n_vectors={self.n_vectors}, dim={self.dim}, "
            f"R={self._n_neighbors}, dtype={self._data_type.value}, "
            f"dist={self._distance.value})"
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