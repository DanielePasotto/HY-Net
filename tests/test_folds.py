import pytest

from hynet import folds

DOC = {"holdout": True, "folds": [{"train": ["A", "B"], "val": ["C"], "test": ["D", "E"]}]}

def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "folds.json"
    folds.save_folds(DOC, str(path))
    assert folds.load_folds(str(path)) == DOC
    assert folds.fold_wells(DOC, 0, "test") == ["D", "E"]
    with pytest.raises(ValueError):
        folds.fold_wells(DOC, 1, "test")

def test_overlapping_sets_are_rejected():
    with pytest.raises(AssertionError):
        folds.check_folds({"holdout": True, "folds": [{"train": ["A"], "val": ["B"], "test": ["A"]}]})
