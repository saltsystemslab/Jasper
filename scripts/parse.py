import csv
import glob
import os
import re


def parse_construction_logs(
    pattern="./results/construct_*.log",
    out_csv="./results/csv/construct_times.csv",
):
    rows = []
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath) as f:
            for line in f:
                m = re.search(r"\[construct\] timing summary \(\d+ batches,\s*([\d.]+) ms total\)", line)
                if m:
                    stem = os.path.splitext(os.path.basename(filepath))[0]
                    name = re.sub(r"^construction_", "", stem)
                    rows.append({"name": name, "total_ms": float(m.group(1))})
                    break

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "total_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {out_csv}")
    return rows


def parse_query_logs(
    pattern="./results/query_*.log",
    out_dir="./results/csv",
):
    BW_RE = re.compile(r"beam_width=(\d+).*throughput=(\d+) QPS")
    RC_RE = re.compile(r"Recall@\d+:\s*([\d.]+)")
    FN_RE = re.compile(r"^query_(.+)_k(\d+)$")

    # Collect rows per dataset
    by_dataset = {}

    for filepath in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        m = FN_RE.match(stem)
        if not m:
            print(f"[WARN] unexpected filename format, skipping: {filepath}")
            continue
        dataset, k = m.group(1), int(m.group(2))

        with open(filepath) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            bw_match = BW_RE.search(lines[i])
            if bw_match:
                rc_match = RC_RE.search(lines[i + 1]) if i + 1 < len(lines) else None
                by_dataset.setdefault(dataset, []).append({
                    "dataset":    dataset,
                    "k":          k,
                    "beam_width": int(bw_match.group(1)),
                    "throughput": int(bw_match.group(2)),
                    "recall":     float(rc_match.group(1)) if rc_match else None,
                })
            i += 1

    os.makedirs(out_dir, exist_ok=True)
    for dataset, rows in by_dataset.items():
        out_csv = os.path.join(out_dir, f"query_{dataset}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "k", "beam_width", "throughput", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} row(s) to {out_csv}")

    return by_dataset


def parse_block_tuning_logs(
    pattern="./results/block_tuning_*.log",
    out_dir="./results/csv",
):
    SECTION_RE = re.compile(
        r"=== (\w+) Beam Search Benchmark \(k=\d+, block_size=(\d+)(?:, tile_size=(\d+))?\) ==="
    )
    BW_RE = re.compile(r"beam_width=(\d+).*throughput=(\d+) QPS")
    RC_RE = re.compile(r"Recall@\d+:\s*([\d.]+)")
    FN_RE = re.compile(r"^block_tuning_(.+)_k(\d+)$")

    # Collect rows per dataset
    by_dataset = {}

    for filepath in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        m = FN_RE.match(stem)
        if not m:
            print(f"[WARN] unexpected filename format, skipping: {filepath}")
            continue
        dataset, k = m.group(1), int(m.group(2))

        with open(filepath) as f:
            lines = f.readlines()

        search_type = None
        block_size = None
        tile_size = None

        i = 0
        while i < len(lines):
            sec = SECTION_RE.search(lines[i])
            if sec:
                search_type = sec.group(1)
                block_size  = int(sec.group(2))
                tile_size   = int(sec.group(3)) if sec.group(3) else 4
                i += 1
                continue

            bw_match = BW_RE.search(lines[i])
            if bw_match and search_type is not None:
                rc_match = RC_RE.search(lines[i + 1]) if i + 1 < len(lines) else None
                by_dataset.setdefault(dataset, []).append({
                    "dataset":     dataset,
                    "k":           k,
                    "search_type": search_type,
                    "block_size":  block_size,
                    "tile_size":   tile_size,
                    "beam_width":  int(bw_match.group(1)),
                    "throughput":  int(bw_match.group(2)),
                    "recall":      float(rc_match.group(1)) if rc_match else None,
                })

            i += 1

    os.makedirs(out_dir, exist_ok=True)
    for dataset, rows in by_dataset.items():
        out_csv = os.path.join(out_dir, f"block_tuning_{dataset}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "k", "search_type", "block_size", "tile_size",
                            "beam_width", "throughput", "recall"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} row(s) to {out_csv}")

    return by_dataset


if __name__ == "__main__":
    parse_construction_logs()
    parse_query_logs()
    parse_block_tuning_logs()