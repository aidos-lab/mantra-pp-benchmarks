from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _options(group):
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


EXPERIMENTS = [e for e in _options("experiment/cy") if e != "base"]
CHAINS = _options("chain")
TASKS = _options("cy_task")


def _compose(experiment, chain, task, *overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="config",
            overrides=[
                f"+experiment=cy/{experiment}",
                f"+chain={chain}",
                f"+cy_task={task}",
                *overrides,
            ],
        )


@pytest.mark.parametrize("chain", CHAINS)
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_cy_experiment_and_chain_compose(experiment, chain):
    cfg = _compose(experiment, chain, TASKS[0])
    assert cfg.dataset.type == "cy"
    assert set(cfg.dataset.transform) == set(cfg.dataset.transform_order)
    if "ect" in cfg.dataset.transform and cfg.model._target_.endswith("ECTViT"):
        # The complexes are 4-dimensional, so every rank means five.
        ect = cfg.dataset.transform.ect
        ranks = 5 if ect.max_rank is None else ect.max_rank + 1
        assert cfg.model.num_ranks == (ranks if ect.per_rank else 1)


@pytest.mark.parametrize("task", TASKS)
def test_cy_task_rows_stratify_classification_on_their_target(task):
    cfg = _compose(EXPERIMENTS[0], CHAINS[0], task)
    if cfg.task.task_type == "classification":
        assert cfg.dataset.use_stratified
        assert cfg.dataset.label_source == cfg.task.target
    else:
        assert not cfg.dataset.use_stratified


def test_command_line_overrides_win():
    cfg = _compose(
        "gcn", "levic", "h21_class", "model.dropout=0.5", "dataset.min_sample_per_class=20"
    )
    assert cfg.model.dropout == 0.5
    assert cfg.dataset.min_sample_per_class == 20
