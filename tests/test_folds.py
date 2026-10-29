import numpy as np
import pytest

from hynet import folds

def wells(n=66, seed=0):
    rng = np.random.default_rng(seed)
    return {f"W{i}": float(rng.uniform(0.05, 0.4)) for i in range(n)}

def test_every_well_is_tested_exactly_once():
    doc = folds.make_folds(wells(), k=5, seed=1)
    folds.check_folds(doc)
    tested = [w for f in doc["folds"] for w in f["test"]]
    assert sorted(tested) == sorted(wells())
    for i, f in enumerate(doc["folds"]):
        assert set(f["val"]) == set(doc["folds"][(i + 1) % 5]["test"])
        assert not set(f["train"]) & (set(f["val"]) | set(f["test"]))

def test_folds_are_balanced_in_size_and_density():
    fractions = wells()
    doc = folds.make_folds(fractions, k=5, seed=1)
    sizes = [len(f["test"]) for f in doc["folds"]]
    assert max(sizes) - min(sizes) <= 1
    density = [np.mean([fractions[w] for w in f["test"]]) for f in doc["folds"]]
    assert max(density) - min(density) < 0.03  # random assignment of 66 wells would typically differ more

def test_reproducible_and_seed_dependent():
    a = folds.make_folds(wells(), seed=3)
    assert a == folds.make_folds(wells(), seed=3)
    assert a["folds"] != folds.make_folds(wells(), seed=4)["folds"]

def test_groups_are_never_split():
    fractions = wells(30)
    groups = {w: f"donor{i // 3}" for i, w in enumerate(sorted(fractions))}  # 10 donors, 3 wells each
    doc = folds.make_folds(fractions, k=5, seed=0, groups=groups)
    folds.check_folds(doc)
    for f in doc["folds"]:
        for part in ("train", "val", "test"):
            donors_here = {groups[w] for w in f[part]}
            for other in set(f) - {part}:
                assert not donors_here & {groups[w] for w in f[other]}

def test_save_load_roundtrip(tmp_path):
    doc = folds.make_folds(wells(20), k=4, seed=0)
    path = tmp_path / "sub" / "folds.json"
    folds.save_folds(doc, str(path))
    assert folds.load_folds(str(path)) == doc
    assert folds.fold_wells(doc, 2, "test") == doc["folds"][2]["test"]
    with pytest.raises(ValueError):
        folds.fold_wells(doc, 9, "test")

def test_too_few_units():
    with pytest.raises(ValueError):
        folds.make_folds({"A": 0.1, "B": 0.2})
