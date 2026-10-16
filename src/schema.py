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
    metrics: Dict[str, MetricMeta] = field(default_factory=dict)

@dataclass
class TransformMeta:
    _target_: str

@dataclass
class DatasetMeta:
    root: str
    version: str = "latest"
    dimension: int = 2
    local_path: str | None = None
    train_parameters: dict = field(default_factory=dict)
    target: str = "name"
    reload: bool = False
    balanced: bool = False
    use_stratified: bool = True

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


