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
from collections import Counter

import hydra
from hydra.core.config_store import ConfigStore

from src.schema import GenerateBarycentricConfig
from src.utils.barycentric_utils import barycentric_subdivision_raw


cs = ConfigStore.instance()
cs.store(name="generate_barycentric_conf", node=GenerateBarycentricConfig)


def subdivide_once(entries):
    out = []
    for entry in entries:
        tri, n_v = barycentric_subdivision_raw(entry["triangulation"])
        new_entry = dict(entry)
        new_entry["triangulation"] = tri
        new_entry["n_vertices"] = n_v
        out.append(new_entry)
    return out


def filter_by_class_count(entries, label_source, min_count):
    """Drop entries whose ``label_source`` value occurs ≤ ``min_count`` times."""
    if min_count <= 0:
        return entries, Counter()
    counts = Counter(e[label_source] for e in entries)
    kept_labels = {lbl for lbl, c in counts.items() if c > min_count}
    filtered = [e for e in entries if e[label_source] in kept_labels]
    return filtered, counts


@hydra.main(version_base=None, config_path="../config", config_name="generate_barycentric_data")
def main(cfg: GenerateBarycentricConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(cfg.input) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} raw triangulations from {cfg.input}")

    if cfg.min_class_count > 0:
        before = len(data)
        data, counts = filter_by_class_count(data, cfg.label_source, cfg.min_class_count)
        kept_classes = sorted({e[cfg.label_source] for e in data})
        print(
            f"[INFO] Class filter ({cfg.label_source} > {cfg.min_class_count}): "
            f"{before} -> {len(data)} entries across {len(kept_classes)} classes"
        )
        print(f"       classes: {kept_classes}")

    current = data
    for level in range(1, cfg.n_levels + 1):
        current = subdivide_once(current)

        filtered = [e for e in current if e["n_vertices"] >= cfg.min_vertices]
        n_dropped = len(current) - len(filtered)

        out_path = os.path.join(cfg.output_dir, f"{cfg.prefix}_bary_{level}.json")
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


if __name__ == "__main__":
    main()
