"""Concurrency-safety test: two threads concurrently append AND delete on one
jasper.Graph, verifying the host-side single-writer lock (graph<>::rw_mutex)
keeps everything consistent through the FFI.

Each mutating call (append -> append_batch, mark_deleted) takes the exclusive
write lock; the id counter is serialized on the same lock. With two threads
hammering both operations we assert:

  * no crash and no deadlock;
  * exact final n_vectors (deletes tombstone slots, they don't remove them);
  * the WHOLE id space [0, n_vectors) is present exactly once -- the ids from
    build() plus every concurrently-appended id, unique and gap-free (a race on
    the monotonic id counter would collide ids or leave gaps);
  * n_tombstoned == number of distinct ids deleted, and n_live == the rest;
  * search never returns a deleted id (delete-by-id excludes it from results);
  * inserts AFTER deletions keep advancing the id counter (deleted ids are not
    recycled) and the new vectors are immediately findable.

Deletion is by stable id: each thread deletes a distinct subset of the ids it
itself appended, so the counts are exact. Run directly:
    python python/tests/test_concurrent_append.py
"""

import threading

import torch

import jasper

DIM = 64
N0 = 20_000          # initial graph size (ids [0, N0))
M = 2_000            # vectors appended per round
D = 500              # ids deleted per round (from a previous round)
ROUNDS = 8           # rounds per thread
N_THREADS = 2


def main() -> None:
    assert torch.cuda.is_available(), "this test needs a CUDA device"

    g0 = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(N0, DIM, dtype=torch.float16, device="cuda", generator=g0)
    g = jasper.Graph.build(base, n_neighbors=64, distance="l2")
    assert g.n_vectors == N0, (g.n_vectors, N0)

    appended: dict[int, torch.Tensor] = {}
    deleted: dict[int, torch.Tensor] = {}
    errors: list[tuple[int, str]] = []
    lock = threading.Lock()
    start = threading.Barrier(N_THREADS)  # release both threads together

    def worker(tid: int) -> None:
        try:
            gen = torch.Generator(device="cuda").manual_seed(100 + tid)
            app_rounds: list[torch.Tensor] = []
            del_rounds: list[torch.Tensor] = []
            start.wait()
            for r in range(ROUNDS):
                v = torch.randn(M, DIM, dtype=torch.float16, device="cuda", generator=gen)
                ids = g.append(v, alpha=1.2).detach().to("cpu")
                app_rounds.append(ids)
                if r >= 1:  # delete D ids appended a round ago (this thread's own, distinct)
                    victims = app_rounds[r - 1][:D].contiguous()
                    g.mark_deleted(victims)
                    del_rounds.append(victims)
            with lock:
                appended[tid] = torch.cat(app_rounds)
                deleted[tid] = (torch.cat(del_rounds) if del_rounds
                                else torch.empty(0, dtype=torch.int64))
        except Exception as e:  # pragma: no cover - reported below
            with lock:
                errors.append((tid, repr(e)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)

    alive = sum(t.is_alive() for t in threads)
    assert alive == 0, f"DEADLOCK: {alive} thread(s) still running after 600s"
    assert not errors, f"worker threads raised: {errors}"

    total_app = N_THREADS * ROUNDS * M
    total_del = N_THREADS * (ROUNDS - 1) * D
    expected_n = N0 + total_app

    # 1) n_vectors is exact -- tombstones do not shrink it.
    assert g.n_vectors == expected_n, f"n_vectors {g.n_vectors} != {expected_n}"

    # 2) The WHOLE id space [0, n_vectors) is covered exactly once: initial ids
    #    [0, N0) from build() plus every appended id, unique and gap-free.
    app_ids = torch.cat([appended[t] for t in appended]).tolist()
    assert len(app_ids) == total_app, (len(app_ids), total_app)
    assert len(app_ids) == len(set(app_ids)), "duplicate appended ids (id-counter race)"
    full = set(range(N0)) | set(app_ids)
    assert full == set(range(expected_n)), "id space is not exactly [0, n_vectors)"
    # Explicit contiguity: the sorted id space is 0, 1, ..., n_vectors-1 with no
    # gaps and no duplicates.
    srt = sorted(full)
    assert (len(srt) == expected_n and srt[0] == 0 and srt[-1] == expected_n - 1
            and all(srt[i] + 1 == srt[i + 1] for i in range(len(srt) - 1))), \
        "id range is not contiguous [0, n_vectors)"

    # 3) Deletion bookkeeping: distinct ids, within the appended range, counted.
    del_ids = torch.cat([deleted[t] for t in deleted]).tolist()
    assert len(del_ids) == len(set(del_ids)) == total_del, "delete count/uniqueness off"
    assert set(del_ids) <= set(range(N0, expected_n)), "deleted ids outside appended range"
    assert g.n_tombstoned == total_del, f"n_tombstoned {g.n_tombstoned} != {total_del}"
    assert g.n_live == expected_n - total_del, f"n_live {g.n_live} != {expected_n - total_del}"

    # 4) Delete-by-id actually excludes from search: no deleted id comes back.
    del_set = set(del_ids)
    gq = torch.Generator(device="cuda").manual_seed(999)
    q = torch.randn(1000, DIM, dtype=torch.float16, device="cuda", generator=gq)
    res_ids, _ = g.search(q, k=10, beam_width=64)
    got = [i for i in res_ids.reshape(-1).tolist() if i >= 0]
    leaked = del_set.intersection(got)
    assert not leaked, f"search returned {len(leaked)} deleted ids"

    # 5) Inserts AFTER deletions are healthy: ids keep advancing monotonically
    #    (deleted ids are NOT recycled), n_vectors grows, tombstones are
    #    untouched, and the freshly inserted vectors are findable.
    K = 5_000
    gk = torch.Generator(device="cuda").manual_seed(555)
    newv = torch.randn(K, DIM, dtype=torch.float16, device="cuda", generator=gk)
    new_ids = g.append(newv, alpha=1.2).detach().to("cpu").tolist()
    assert new_ids == list(range(expected_n, expected_n + K)), \
        "post-deletion append ids are not fresh/contiguous"
    assert not (set(new_ids) & del_set), "post-deletion append recycled a deleted id"
    assert g.n_vectors == expected_n + K, (g.n_vectors, expected_n + K)
    assert g.n_tombstoned == total_del, "a plain append changed the tombstone count"
    # each new vector should find itself at rank 1
    nres_ids, _ = g.search(newv, k=10, beam_width=64)
    top1 = nres_ids[:, 0].tolist()
    found = sum(1 for got1, nid in zip(top1, new_ids) if got1 == nid)
    recall1 = found / K
    assert recall1 >= 0.95, f"post-deletion inserts self-recall@1={recall1:.4f} too low"

    print(
        f"PASS: {N_THREADS} threads concurrent append+delete -> "
        f"whole id space [0,{expected_n}) unique+contiguous, "
        f"n_tombstoned={total_del}, n_live={expected_n - total_del}, 0 deleted in search; "
        f"post-delete +{K} fresh ids [{expected_n},{expected_n + K}) "
        f"self-recall@1={recall1:.4f}, final n_vectors={g.n_vectors}"
    )


if __name__ == "__main__":
    main()
