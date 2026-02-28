"""Jasper — GPU graph-based nearest neighbor search."""

from pathlib import Path
from enum import Enum

import torch
import tvm_ffi

# ── Load the compiled shared library ────────────────────────────
_lib_path = str(Path(__file__).parent / "lib" / "libjasper_ffi.so")
_mod: tvm_ffi.Module = tvm_ffi.load_module(_lib_path)


# ── Config enums ────────────────────────────────────────────────
class DistanceFunc(str, Enum):
    L2 = "l2"
    INNER_PRODUCT = "ip"


class DataType(str, Enum):
    FLOAT32 = "f32"
    UINT8 = "u8"


# ── Config registry (must match JASPER_FOR_EACH_CONFIG in C++) ──
# Maps (data_type, n_neighbors, distance_func) → config_id string
_CONFIGS: dict[tuple[DataType, int, DistanceFunc], str] = {
    (DataType.FLOAT32, 32,  DistanceFunc.L2):             "f32_r32_l2",
    (DataType.FLOAT32, 64,  DistanceFunc.L2):             "f32_r64_l2",
    (DataType.FLOAT32, 128, DistanceFunc.L2):             "f32_r128_l2",
    (DataType.FLOAT32, 32,  DistanceFunc.INNER_PRODUCT):  "f32_r32_ip",
    (DataType.FLOAT32, 64,  DistanceFunc.INNER_PRODUCT):  "f32_r64_ip",
    (DataType.UINT8,   32,  DistanceFunc.L2):             "u8_r32_l2",
    (DataType.UINT8,   64,  DistanceFunc.L2):             "u8_r64_l2",
}

# Torch dtype mapping
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
            getattr(_mod, f"jasper_search_{config_id}"),
        )
    return _fn_cache[config_id]


# ── Graph class ─────────────────────────────────────────────────
class Graph:
    """
    A GPU-resident graph index for nearest neighbor search.

    Usage:
        g = jasper.Graph("sift1m.graph", dim=128, n_neighbors=32)
        indices, distances = g.search(queries, k=10)
    """

    def __init__(
        self,
        path: str,
        dim: int,
        n_neighbors: int = 32,
        data_type: DataType | str = DataType.FLOAT32,
        distance: DistanceFunc | str = DistanceFunc.L2,
    ):
        """
        Load a graph from a binary file into GPU memory.

        Args:
            path:         Path to the graph binary file.
            dim:          Dimensionality of the vectors.
            n_neighbors:  Max neighbors per node (must match file).
            data_type:    Vector data type: "f32" or "u8".
            distance:     Distance function: "l2" or "ip".
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)
        if isinstance(distance, str):
            distance = DistanceFunc(distance)

        self._config_id = _resolve_config(data_type, n_neighbors, distance)
        self._data_type = data_type
        self._distance = distance
        self._n_neighbors = n_neighbors
        self._dim = dim
        self._torch_dtype = _DTYPE_MAP[data_type]

        load_fn, self._search_fn = _get_fns(self._config_id)
        self._handle: int | None = load_fn(path, dim)

        self._free_fn = _free_graph_fn

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run beam search on this graph.

        Args:
            queries:    CUDA tensor of shape [n_queries, dim]
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
            k, beam_width, limit
        )

        return out_indices, out_distances

    def free(self):
      if self._handle is not None:
        self._free_fn(self._handle)
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


# ── Usage ───────────────────────────────────────────────────────
#
# import jasper
#
# # Float32 vectors, 32 neighbors, L2 distance
# g = jasper.Graph("sift1m.graph", dim=128, n_neighbors=32)
# print(g)  # Graph(n_vectors=10000000, dim=128, R=32, dtype=f32, dist=l2)
#
# queries = torch.randn(1000, 128, device="cuda")
# indices, distances = g.search(queries, k=10, beam_width=64)
#
# # Uint8 vectors, 64 neighbors, L2
# g2 = jasper.Graph("bigann.graph", dim=128, n_neighbors=64, data_type="u8")
#
# # Inner product
# g3 = jasper.Graph("emb.graph", dim=768, n_neighbors=32, distance="ip")