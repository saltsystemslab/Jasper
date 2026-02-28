"""Jasper — GPU graph-based nearest neighbor search."""

from pathlib import Path

import torch
import tvm_ffi

# ── Load the compiled shared library ────────────────────────────
_lib_path = str(Path(__file__).parent / "lib" / "libjasper_ffi.so")
_mod: tvm_ffi.Module = tvm_ffi.load_module(_lib_path)

_load_graph_fn = _mod.jasper_load_graph
_save_graph_fn = _mod.jasper_save_graph
_search_fn = _mod.jasper_search


# ── State ───────────────────────────────────────────────────────
_graph_dim: int | None = None


# ── Public API ──────────────────────────────────────────────────
def load_graph(path: str, dim: int) -> None:
    """
    Load a graph from a binary file into GPU memory.

    Args:
        path: Path to the graph binary file.
        dim:  Dimensionality of the vectors in the graph.
    """
    global _graph_dim
    _load_graph_fn(path, dim)
    _graph_dim = dim


def save_graph(path: str) -> None:
    """Save the currently loaded graph to a binary file."""
    _save_graph_fn(path)


def search(
    queries: torch.Tensor,
    k: int = 10,
    beam_width: int = 64,
    limit: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run beam search on the loaded graph.

    Args:
        queries:    float32 CUDA tensor of shape [n_queries, dim]
        k:          Number of nearest neighbors to return.
        beam_width: Search beam width (higher = more accurate, slower).
        limit:      Max iterations per query.

    Returns:
        indices:   int32  tensor [n_queries, k] — neighbor IDs
        distances: float32 tensor [n_queries, k] — distances to neighbors
    """
    if _graph_dim is None:
        raise RuntimeError("No graph loaded. Call jasper.load_graph() first.")

    if not queries.is_cuda:
        raise ValueError("queries must be a CUDA tensor")
    if queries.dtype != torch.float32:
        raise ValueError("queries must be float32")
    if queries.ndim != 2:
        raise ValueError(f"queries must be 2D [n_queries, dim], got {queries.ndim}D")
    if queries.size(1) != _graph_dim:
        raise ValueError(
            f"query dim ({queries.size(1)}) doesn't match graph dim ({_graph_dim})"
        )
    # Ensure contiguous layout for zero-copy
    queries = queries.contiguous()

    n_queries = queries.size(0)

    # Pre-allocate output tensors on the same device
    out_indices = torch.empty(
        n_queries, k, dtype=torch.int32, device=queries.device
    )
    out_distances = torch.empty(
        n_queries, k, dtype=torch.float32, device=queries.device
    )

    # Call into C++ — tensors pass zero-copy via DLPack
    _search_fn(queries, out_indices, out_distances, k, beam_width, limit)

    return out_indices, out_distances


# ── Usage ───────────────────────────────────────────────────────
#
# import jasper
#
# # Load a graph (vectors are 128-dimensional)
# jasper.load_graph("sift1m.graph", dim=128)
#
# # Search
# queries = torch.randn(1000, 128, device="cuda", dtype=torch.float32)
# indices, distances = jasper.search(queries, k=10, beam_width=64)
#
# print(indices.shape)     # torch.Size([1000, 10])
# print(distances.shape)   # torch.Size([1000, 10])
#
# # Save
# jasper.save_graph("sift1m_copy.graph")