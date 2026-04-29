"""Hydra-configured barycentric subdivision generator.

Applies repeated barycentric subdivision to every entry in a Mantra-style
raw triangulation JSON file and writes one output JSON per subdivision
level. Entries whose subdivided vertex count is below ``min_vertices``
are dropped from the output (but kept in the running list so later levels
are computed from the full intermediate).

Optionally applies a pre-subdivision class-count filter
(``min_class_count``) so that training and the precomputed bary caches
see the same class set in the same first-seen order — this mirrors
``filter_low_ocurrence_labels`` in the training pipeline but pushes it
upstream, which makes ``num_classes`` and the ``label_to_index`` mapping
deterministic across processes.

Example
-------
    python src/generate_barycentric_data.py \\
        input=data/mantra/2D/raw/2_manifolds.json \\
        output_dir=/tmp/mantra_bary \\
        prefix=2_manifolds \\
        n_levels=4 \\
        min_vertices=16 \\
        label_source=name \\
        min_class_count=100
"""

import json
import os
import random
from collections import Counter, defaultdict

import hydra
from hydra.core.config_store import ConfigStore

from src.schema import GenerateBarycentricConfig
from src.utils.barycentric_utils import (
    barycentric_subdivision_raw,
    stellar_subdivision_raw,
    barycentric_stellar_graded
)


cs = ConfigStore.instance()
cs.store(name="generate_barycentric_conf", node=GenerateBarycentricConfig)


_SUBDIVISION_FUNCTIONS = {
    "stellar": stellar_subdivision_raw,
    "full_barycentric": barycentric_subdivision_raw,
    "graded": barycentric_stellar_graded
}


def _subdivide_entry(entry, mode, fraction=1.0, rng=None):
    fn = _SUBDIVISION_FUNCTIONS[mode]
    if mode == "stellar":
        tri, n_v = fn(entry["triangulation"], fraction=fraction, rng=rng)
    if mode == "graded":
        tri, n_v = fn(entry['triangulation'], over_vrtx_cnt=fraction, rng=rng)
    else:
        tri, n_v = fn(entry["triangulation"])
    new_entry = dict(entry)
    new_entry["triangulation"] = tri
    new_entry["n_vertices"] = n_v
    return new_entry


def subdivide_once(entries, mode, fraction=1.0, rng=None):
    return [_subdivide_entry(e, mode, fraction=fraction, rng=rng) for e in entries]


def _level_tag(level):
    """Format a level for the output filename: integers as '1', fractions as '0.5'."""
    return f"{level:g}"


def print_statistics(dataset, label_source="name"):
    """Print per-class n_vertices statistics for a list of entries."""
    class_entries = defaultdict(list)
    for entry in dataset:
        class_entries[entry[label_source]].append(entry)

    print(
        f"{'Class':<30} {'Count':>8} {'Min V':>8} {'Mean V':>8} {'Max V':>8}"
    )
    print("-" * 64)
    for name in sorted(class_entries.keys()):
        entries = class_entries[name]
        nverts = [e["n_vertices"] for e in entries]
        mean_v = sum(nverts) / len(nverts)
        print(
            f"{str(name):<30} {len(entries):>8} "
            f"{min(nverts):>8} {mean_v:>8.2f} {max(nverts):>8}"
        )
    print(f"\nTotal: {len(dataset)}")


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs ≤ ``min_count`` times."""
    if min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts


@hydra.main(version_base=None, config_path="config", config_name="generate_barycentric_data")
def main(cfg: GenerateBarycentricConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(cfg.input) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} raw triangulations from {cfg.input}")

    if cfg.mode not in _SUBDIVISION_FUNCTIONS:
        raise ValueError(
            f"Unknown subdivision mode {cfg.mode!r}; expected one of "
            f"{sorted(_SUBDIVISION_FUNCTIONS)}"
        )
    print(f"[INFO] Subdivision mode: {cfg.mode}")

    nl = float(cfg.n_levels)
    if nl <= 0:
        raise ValueError(f"n_levels must be > 0, got {nl}")
    fractional = nl < 1.0
    if fractional and cfg.mode != "stellar":
        raise ValueError(
            "Fractional n_levels (partial subdivision) only supported in mode='stellar'; "
            "full_barycentric breaks the SC condition under partial application."
        )
    n_full_levels = 0 if fractional else int(round(nl))
    if not fractional and abs(nl - n_full_levels) > 1e-9:
        raise ValueError(
            f"n_levels >= 1 must be integer-valued (got {nl}); fractional values "
            f"are only allowed in (0, 1)."
        )
    fraction = nl if fractional else 1.0
    print(
        f"[INFO] n_levels={nl} (fractional={fractional}, fraction={fraction}, "
        f"full_levels={n_full_levels}, seed={cfg.seed})"
    )

    if cfg.min_class_count > 0:
        before = len(data)
        data, counts = filter_by_class_count(data, cfg.label_source, cfg.min_class_count)
        kept_classes = sorted({e[cfg.label_source] for e in data})
        print(
            f"[INFO] Class filter ({cfg.label_source} > {cfg.min_class_count}): "
            f"{before} -> {len(data)} entries across {len(kept_classes)} classes"
        )
        print(f"       classes: {kept_classes}")

    rng = random.Random(cfg.seed)
    first_level_tag = _level_tag(fraction if fractional else 1)

    if cfg.n_smallest > 0:
        per_class = defaultdict(list)
        # name -> List of triangulations with this manifold type
        for entry in data:
            per_class[entry[cfg.label_source]].append(entry)

        cohort = []
        n_dropped = 0
        for label in sorted(per_class.keys()):
            # Sort incide per number of vertices
            sorted_entries = sorted(per_class[label], key=lambda e: e["n_vertices"], ascending=(cfg.mode != 'graded')) # Descending if it's graded
            kept = 0

            for entry in sorted_entries:
                sub = _subdivide_entry(entry, cfg.mode, fraction=fraction, rng=rng)
                if sub["n_vertices"] >= cfg.min_vertices:
                    cohort.append(sub)
                    kept += 1
                    if kept == cfg.n_smallest:
                        break
                else:
                    n_dropped += 1
            if kept < cfg.n_smallest:
                print(
                    f"[WARN] Class {label!r}: only {kept} valid samples found "
                    f"(requested {cfg.n_smallest}); pool exhausted."
                )

        out_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_bary_{first_level_tag}.json")
        with open(out_path, "w") as f:
            json.dump(cohort, f)
        ns = [e["n_vertices"] for e in cohort]
        print(
            f"[INFO] Level {first_level_tag}: kept {len(cohort)} smallest valid "
            f"({cfg.n_smallest}/class, skipped {n_dropped} below "
            f"{cfg.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(cohort, cfg.label_source)

        if fractional:
            return

        current = cohort
        for level in range(2, n_full_levels + 1):
            current = subdivide_once(current, cfg.mode)
            below = [e for e in current if e["n_vertices"] < cfg.min_vertices]
            if below:
                print(
                    f"[WARN] Level {level}: {len(below)} entries below "
                    f"{cfg.min_vertices} vertices after subdivision (unexpected)."
                )
            out_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_bary_{_level_tag(level)}.json")
            with open(out_path, "w") as f:
                json.dump(current, f)
            ns = [e["n_vertices"] for e in current]
            print(f"[INFO] Level {level}: kept {len(current)} -> {out_path}")
            if ns:
                print(
                    f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                    f"mean={sum(ns) / len(ns):.1f}"
                )
            print_statistics(current, cfg.label_source)
        return

    if fractional:
        current = subdivide_once(data, cfg.mode, fraction=fraction, rng=rng)
        filtered = [e for e in current if e["n_vertices"] >= cfg.min_vertices]
        n_dropped = len(current) - len(filtered)
        out_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_bary_{first_level_tag}.json")
        with open(out_path, "w") as f:
            json.dump(filtered, f)
        ns = [e["n_vertices"] for e in filtered]
        print(
            f"[INFO] Level {first_level_tag}: kept {len(filtered)} "
            f"(dropped {n_dropped} below {cfg.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(filtered, cfg.label_source)
        return

    current = data
    for level in range(1, n_full_levels + 1):
        current = subdivide_once(current, cfg.mode)

        filtered = [e for e in current if e["n_vertices"] >= cfg.min_vertices]
        n_dropped = len(current) - len(filtered)

        out_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_bary_{_level_tag(level)}.json")
        with open(out_path, "w") as f:
            json.dump(filtered, f)

        ns = [e["n_vertices"] for e in filtered]
        print(
            f"[INFO] Level {level}: kept {len(filtered)} "
            f"(dropped {n_dropped} below {cfg.min_vertices} vertices) -> {out_path}"
        )
        if ns:
            print(
                f"       n_vertices: min={min(ns)}, max={max(ns)}, "
                f"mean={sum(ns) / len(ns):.1f}"
            )
        print_statistics(filtered, cfg.label_source)


if __name__ == "__main__":
    main()
