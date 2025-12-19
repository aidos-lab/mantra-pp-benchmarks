from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.data import Data
from omegaconf import DictConfig, OmegaConf
from mantra.datasets import ManifoldTriangulations
from mantra.transforms import MomentCurveEmbedding, SelectAttributes, SelectFeatures
import hydra
from hydra.core.config_store import ConfigStore
from src.schema import RootConfig

cs = ConfigStore().instance()
cs.store(name="config", node=RootConfig)




@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):


    dataset_config = cfg.dataset
    transform_dict = dataset_config.transforms
    transform_order = dataset_config.transform_order

    transforms = [
        hydra.utils.instantiate(transform) for _, transform in transform_dict.items()
    ]
    transforms = Compose(transforms)

    dataset = ManifoldTriangulations(
        root=dataset_config.root,
        dimension=dataset_config.dimension,
        version=dataset_config.version,
        pre_transform=transforms
    )
    model = hydra.utils.instantiate(cfg.model)


    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
