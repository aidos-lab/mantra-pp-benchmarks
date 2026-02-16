from dataclasses import dataclass, field

from typing import Dict, List, Any


@dataclass
class TrainerMeta:
    learning_rate: float
    weight_decay: float
    max_epoch: int
    early_stopping: int
@dataclass
class MetricMeta:
    name: str

@dataclass
class LossMeta:
    name: str
@dataclass
class TaskMeta:
    name: str
    loss: LossMeta
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
    
    transforms: Dict[str, Any] = field(default_factory=dict)
    transform_order: List[str] = field(default_factory=list)

@dataclass
class ModelMeta:
    _target_: str
    hidden_channels: int = 16
    out_channels: int = 2
    num_node_features: int = 2
    layers: int = 2
    residuals: bool = True
    batch_norm: bool = True

@dataclass
class LoggerMeta:
    name: str
    project: str

@dataclass
class RootConfig:
    logger: LoggerMeta
    dataset: DatasetMeta
    model: ModelMeta
    task: TaskMeta
    trainer: TrainerMeta

    train: bool = True
    seed: int = 42

