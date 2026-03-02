from dataclasses import dataclass, field

from typing import Dict, List, Any


@dataclass
class TrainerMeta:
    learning_rate: float
    weight_decay: float
    max_epoch: int
    early_stopping: int
    batch_size: int
    use_stratified: bool
    device: str

@dataclass
class MetricMeta:
    _target_: str

@dataclass
class LossMeta:
    __target__: str
    name: str

@dataclass
class TaskMeta:
    name: str
    loss: LossMeta
    target: str
    task_type: str
    imbalanced: bool = False
    metrics: List[MetricMeta] = field(default_factory=list)

@dataclass
class TransformMeta:
    _target_: str

@dataclass
class DatasetMeta:
    root: str
    version: str = "latest"
    dimension: int = 2
    train_parameters: dict = field(default_factory=dict)
    target: str = "name"
    reload: bool = False
    
    transform: Dict[str, Any] = field(default_factory=dict)
    transform_order: List[str] = field(default_factory=list)

    num_features: int | str = 'auto'

@dataclass
class ModelMeta:
    _target_: str
    hidden_channels: int = 16
    out_channels: int = 2
    num_node_features: int = 2
    layers: int = 2
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

    train: bool = True
    seed: int = 42

