from __future__ import annotations

from ezscreen.backends import engines


def test_default_engine_is_implemented():
    prof = engines.get(engines.DEFAULT_ENGINE)
    assert prof.implemented
    assert prof.requires_box


def test_unknown_engine_falls_back_to_default():
    assert engines.get("does-not-exist").key == engines.DEFAULT_ENGINE


def test_gnina_supports_cnn_and_vina_family():
    g = engines.get("gnina")
    assert g.supports_cnn
    # ranks by Vina affinity (CNN columns are supplementary), so the merger is unchanged
    assert g.native_score_type == "vina_kcal_mol"
    assert "vinardo" in g.scoring_functions


def test_unidock_is_vina_family_without_cnn():
    u = engines.get("unidock")
    assert not u.supports_cnn
    assert u.native_score_type == "vina_kcal_mol"
    assert engines.default_scoring("unidock") == "vina"


def test_implemented_engines_are_a_subset():
    impl = {e.key for e in engines.implemented_engines()}
    everything = {e.key for e in engines.all_engines()}
    # UniDock-Pro's structure-based docking == UniDock, so it is not a distinct
    # engine; its value (LBVS/hybrid) is a future feature.
    assert impl == {"unidock", "gnina"}
    assert impl < everything  # smina / diffdock / unidock-pro declared but not offered


def test_dockable_engines_have_a_template():
    dockable = {e.key for e in engines.dockable_engines()}
    assert dockable == {"unidock", "gnina"}
    for e in engines.dockable_engines():
        assert e.kaggle_template


def test_diffdock_is_box_free():
    d = engines.get("diffdock")
    assert not d.requires_box
    assert d.scoring_functions == ()


def test_scoring_function_membership():
    assert engines.supports_scoring("unidock", "vinardo")
    assert not engines.supports_scoring("unidock", "cnn")
    assert not engines.supports_scoring("autodock-gpu", "vina")
