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
    scheduler_patience: int = 10
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
    # 'graph': one target per sample; 'node': one per vertex; 'triangle': one per triangle.
    task_level: str = "graph"
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
    # Attribute whose value groups samples that must share a split, e.g.
    # `polytope_id` for the phases of one polytope (cy only).
    group_by: str | None = None
    # Processes building the cy cache; None uses every available CPU.
    pre_transform_workers: int | None = None

    seed: int = 42
    transform: Dict[str, TransformMeta] = field(default_factory=dict)
    transform_order: List[str] = field(default_factory=list)
    # Subdivisions of the test split, see the `ood` config group.
    ood_evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
    pool_method: str = "mean"
    task_level: str = "graph"
    # Hidden layers of the readout after pooling or the triangle sum; 0 is a linear layer.
    head_layers: int = 0

@dataclass
class LoggerMeta:
    name: str
    project: str
    entity: str

@dataclass
class RootConfig:
    logger: LoggerMeta
    model: ModelMeta
    dataset: DatasetMeta
    task: TaskMeta
    trainer: TrainerMeta

    checkpoint_path: str

    train: bool = True
    ood: bool = False
    torch_seed: int = 42
    compile: bool = False


