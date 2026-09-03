from dataclasses import dataclass, field

from typing import Dict, List, Any


@dataclass
class TrainerMeta:
    learning_rate: float
    weight_decay: float
    max_epoch: int
    early_stopping: int
    batch_size: int
    device: str
    num_workers: int = 0
    pin_memory: bool = True

@dataclass
class MetricMeta:
    _target_: str

@dataclass
class LossMeta:
    _target_: str = ""

@dataclass
class TaskMeta:
    name: str
    loss: LossMeta
    target: str
    task_type: str
    # 'graph': one target per sample; 'node': one target per vertex,
    # supervised where data.node_mask is set (see select_supervised).
    task_level: str = "graph"
    # Node-level only: exclude the first vertex from the supervision.
    mask_first: bool = False
    metrics: Dict[str, MetricMeta] = field(default_factory=dict)

@dataclass
class TransformMeta:
    _target_: str

@dataclass
class DatasetMeta:
    root: str
    # 'mantra' (JSON manifold triangulations) or 'cy' (parquet
    # Calabi-Yau star triangulations; requires the mantra cy branch).
    type: str = "mantra"
    # Only process the first `limit` samples (cy only).
    limit: int | None = None
    version: str = "latest"
    dimension: int = 2
    local_path: str | None = None
    train_parameters: dict = field(default_factory=dict)
    target: str = "name"
    reload: bool = False
    balanced: bool = False
    use_stratified: bool = True
    # Attribute whose values stratify / filter the splits (cy only).
    label_source: str | None = None

    transform: Dict[str, TransformMeta] = field(default_factory=dict)
    transform_order: List[str] = field(default_factory=list)

@dataclass
class ModelMeta:
    _target_: str
    hidden_channels: int = 16
    out_channels: int = 2
    in_channels: int = 2
    n_layers: int = 2
    residuals: bool = True
    batch_norm: bool = True
    dropout: float = 0.2
    # Graph readout: 'mean', or 'sum' to keep it extensive (targets
    # that scale with the size of the complex).
    pool_method: str = "mean"
    # Set from task.task_level by create_model; 'node' skips the graph
    # readout and emits one output per vertex.
    task_level: str = "graph"

@dataclass
class LoggerMeta:
    name: str
    project: str
    entity: str

@dataclass
class GenerateBarycentricConfig:
    input: str
    output_dir: str
    prefix: str
    n_levels: float = 1.0
    min_vertices: int = 16
    label_source: str = "name"
    min_class_count: int = 100
    n_smallest: int = 100
    mode: str = "stellar"
    seed: int = 42


@dataclass
class EvalBarycentricConfig:
    checkpoint_dir: str
    bary_dataset_root: str
    bary_local_path: str
    bary_level: float = 1.0
    # Must match the `dataset.balanced` that precompute_barycentric.py was
    # invoked with when the cache was built.
    bary_balanced: bool = False
    device: str = "cpu"
    output: str | None = None
    seed: int = 42

@dataclass
class RootConfig:
    logger: LoggerMeta
    model: ModelMeta
    dataset: DatasetMeta
    task: TaskMeta
    trainer: TrainerMeta

    checkpoint_path: str

    train: bool = True
    seed: int = 42
    compile: bool = False


