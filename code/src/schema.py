from dataclasses import dataclass, field

from typing import Dict, List, Any


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
    name: str = "wandb"


@dataclass
class RootConfig:
    logger: LoggerMeta
    dataset: DatasetMeta
    model: ModelMeta

    train: bool = True
    seed: int = 42

