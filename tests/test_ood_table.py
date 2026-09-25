"""The OOD table: several subdivisions evaluated in one run."""

from omegaconf import OmegaConf

from main import ood_evaluations


def make_cfg(division_type="none", subdivision=None, table=None):
    return OmegaConf.create(
        {
            "dataset": {
                "division_type": division_type,
                "subdivision": subdivision or {},
                "ood_evaluations": table or {},
            }
        }
    )


def test_no_subdivision_gives_no_ood_split():
    assert ood_evaluations(make_cfg()) == []


def test_single_split_keeps_the_plain_prefix():
    """The split inherits division_type from the shared parameters."""
    cfg = make_cfg("graded", {"graded_vertex_number": 16})

    assert ood_evaluations(cfg) == [
        ("ood", {"division_type": "graded", "graded_vertex_number": 16})
    ]


def test_table_yields_one_named_split_per_entry():
    cfg = make_cfg(
        "graded",
        {"graded_vertex_number": 16},
        {
            "stellar075": {
                "division_type": "stellar",
                "subdivision": {"fraction": 0.75},
            },
            "bary1": {
                "division_type": "barycentric",
                "subdivision": {"round": 1},
            },
        },
    )

    assert ood_evaluations(cfg) == [
        ("ood_stellar075", {"division_type": "stellar", "fraction": 0.75}),
        ("ood_bary1", {"division_type": "barycentric", "round": 1}),
    ]


def test_table_entry_without_arguments_is_allowed():
    cfg = make_cfg(table={"plain": {"division_type": "none"}})

    assert ood_evaluations(cfg) == [("ood_plain", {"division_type": "none"})]
