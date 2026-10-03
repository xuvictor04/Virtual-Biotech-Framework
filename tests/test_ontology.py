"""Cell Ontology harmonisation with a mini OBO fixture (offline, no optional deps)."""

from pathlib import Path

import pandas as pd
import pytest

from vbt.analysis import ontology as onto

OBO = Path(__file__).parent / "fixtures" / "mini_cl.obo"


@pytest.fixture(scope="module")
def cl():
    return onto.load_cell_ontology(OBO, engine="builtin")


def test_parse_and_depth(cl):
    assert "CL:0009999" not in cl and "CL:0009999" in cl.obsolete
    assert cl.name("CL:0000057") == "fibroblast"
    assert cl.depth("CL:0000000") == 0 and cl.depth("CL:0000057") == 3 and cl.depth("CL:0002553") == 4
    assert "CL:0000499" in cl.ancestors("CL:0002553")


def test_level1_and_level3(cl):
    t = cl.map_terms(["CL:0002553", "CL:0000186", "CL:0000625", "CL:0000235", "CL:0002063", "CL:0000115",
                      "CL:0000499", "CL:9999999"]).set_index("term_id")
    assert t.loc["CL:0002553", "level1"] == "stromal" and t.loc["CL:0002553", "level3"] == "fibroblast"
    assert t.loc["CL:0000186", "level3"] == "fibroblast"  # fibroblast subtypes pool at Level 3
    assert t.loc["CL:0000625", "level1"] == "immune" and t.loc["CL:0000625", "level3"] == "T cell"
    assert t.loc["CL:0000235", "level1"] == "immune" and t.loc["CL:0000235", "level3"] == "macrophage"
    assert t.loc["CL:0002063", "level1"] == "epithelial"
    assert t.loc["CL:0000115", "level1"] == "endothelial"
    assert t.loc["CL:0000499", "level3"] == "stromal cell"  # shallower than 3 -> itself
    assert pd.isna(t.loc["CL:9999999", "level1"])
    # depth-based Level 1 when no anchors are used
    assert cl.level_label("CL:0002553", 1, anchors={}) == "native cell"
    # custom anchors
    assert cl.level_label("CL:0000625", 2, anchors={2: [("CL:0000084", "T")]}) == "T"


def test_add_level_labels(cl):
    obs = pd.DataFrame({"cell_type_ontology_term_id": ["CL:0002553", "CL:0000625", "CL:7777777"],
                        "cell_type": ["fibroblast of lung", "CD8 T", "mystery"]})
    onto.add_level_labels(obs, cl)
    assert obs["cell_type_level1"].tolist() == ["stromal", "immune", "mystery"]
    assert obs["cell_type_level3"].tolist() == ["fibroblast", "T cell", "mystery"]


def test_pronto_engine_agrees(cl):
    pytest.importorskip("pronto")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = onto.load_cell_ontology(OBO, engine="pronto")
    terms = ["CL:0002553", "CL:0000625", "CL:0000115"]
    assert p.map_terms(terms).equals(cl.map_terms(terms))


def test_load_requires_path(monkeypatch):
    monkeypatch.delenv("VBT_CL_OBO", raising=False)
    with pytest.raises(FileNotFoundError):
        onto.load_cell_ontology()
    monkeypatch.setenv("VBT_CL_OBO", str(OBO))
    assert onto.load_cell_ontology(engine="builtin").name("CL:0000738") == "leukocyte"
