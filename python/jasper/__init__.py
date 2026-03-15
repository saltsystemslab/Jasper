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
    FLOAT32 = "f32"
    UINT8 = "uint8"


# ── Config registry (must match JASPER_FOR_EACH_CONFIG in C++) ──
_CONFIGS: dict[tuple[DataType, int, DistanceFunc], str] = {
    (DataType.FLOAT32, 32,  DistanceFunc.L2):             "f32_r32_l2",
    (DataType.FLOAT32, 64,  DistanceFunc.L2):             "f32_r64_l2",
    (DataType.FLOAT32, 128, DistanceFunc.L2):             "f32_r128_l2",
    (DataType.FLOAT32, 32,  DistanceFunc.INNER_PRODUCT):  "f32_r32_ip",
    (DataType.FLOAT32, 64,  DistanceFunc.INNER_PRODUCT):  "f32_r64_ip",
    (DataType.UINT8,   32,  DistanceFunc.L2):             "u8_r32_l2",
    (DataType.UINT8,   64,  DistanceFunc.L2):             "u8_r64_l2",
}

_DTYPE_MAP = {
    DataType.FLOAT32: torch.float32,
    DataType.UINT8: torch.uint8,
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
_get_dim_fn = _mod.jasper_get_dim


def _get_fns(config_id: str):
    if config_id not in _fn_cache:
        _fn_cache[config_id] = (
            getattr(_mod, f"jasper_load_{config_id}"),
            getattr(_mod, f"jasper_construct_{config_id}"),
            getattr(_mod, f"jasper_search_{config_id}"),
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
        _, _, self._search_fn = _get_fns(config_id)

    @classmethod
    def load(
        cls,
        path: str,
        dim: int,
        n_neighbors: int = 32,
        data_type: DataType | str = DataType.FLOAT32,
        distance: DistanceFunc | str = DistanceFunc.L2,
    ) -> "Graph":
        """
        Load a graph from a binary file into GPU memory.

        Args:
            path:         Path to the graph binary file.
            dim:          Dimensionality of the vectors.
            n_neighbors:  Max neighbors per node (must match file).
            data_type:    Vector data type: "f32" or "uint8".
            distance:     Distance function: "l2" or "ip".
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        config_id = _resolve_config(data_type, n_neighbors, distance)
        load_fn, _, _ = _get_fns(config_id)
        handle = load_fn(path, dim)

        return cls(handle, config_id, data_type, distance, n_neighbors, dim)

    @classmethod
    def build(
        cls,
        vectors: torch.Tensor,
        n_neighbors: int = 32,
        distance: DistanceFunc | str = DistanceFunc.L2,
        alpha: float = 1.2,
        workspace_budget: str = "10GB",
    ) -> "Graph":
        """
        Construct a graph index from vectors on GPU.

        Args:
            vectors:        CUDA tensor of shape [n_vectors, dim].
            n_neighbors:    Max neighbors per node (R).
            distance:       Distance function: "l2" or "ip".
            alpha:          Pruning factor (1.0 = strict, >1.0 = long hops).
            max_batch_size: Max vectors processed per batch during construction.

        Returns:
            A Graph ready for search.
        """
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        workspace_budget_bytes = parse_storage_size(workspace_budget)

        if not vectors.is_cuda:
            raise ValueError("vectors must be a CUDA tensor")
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got {vectors.ndim}D")

        vectors = vectors.contiguous()
        dim = vectors.size(1)

        # Infer data type from tensor dtype
        if vectors.dtype == torch.float32:
            data_type = DataType.FLOAT32
        elif vectors.dtype == torch.uint8:
            data_type = DataType.UINT8
        else:
            raise ValueError(
                f"Unsupported tensor dtype: {vectors.dtype}. "
                f"Use torch.float32 or torch.uint8."
            )

        config_id = _resolve_config(data_type, n_neighbors, distance)
        _, construct_fn, _ = _get_fns(config_id)
        handle = construct_fn(vectors, dim, alpha, workspace_budget_bytes)

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
        limit: int = 512,
        print_throughput: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search on this graph.

        Args:
            queries:    CUDA tensor of shape [n_queries, dim].
            k:          Number of nearest neighbors to return.
            beam_width: Search beam width.
            limit:      Max iterations per query.

        Returns:
            indices:   int32  tensor [n_queries, k]
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
        
        self._search_fn(
            self._handle, queries, out_indices, out_distances,
            k, beam_width, limit, print_throughput
        )

        return out_indices, out_distances

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

def read_bin(path: str, dtype: str = "float32", max_vectors: int = 0) -> torch.Tensor:
    np_dtype = np.float32 if dtype == "float32" else np.uint8
    torch_dtype = torch.float32 if dtype == "float32" else torch.uint8

    with open(path, "rb") as f:
        n_vectors, dim = struct.unpack("II", f.read(8))
        if max_vectors > 0:
            n_vectors = min(n_vectors, max_vectors)
        nbytes = n_vectors * dim * np.dtype(np_dtype).itemsize
        data = np.frombuffer(f.read(nbytes), dtype=np_dtype).reshape(n_vectors, dim).copy()

    return torch.from_numpy(data).to(device="cuda", dtype=torch_dtype)

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
    recall = sum(
        len(set(result_indices[i].tolist()) & set(gt[i].tolist()))
        for i in range(n_queries)
    ) / (n_queries * k)
    print(f"Recall@10: {recall:.3f}")
    return recall

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