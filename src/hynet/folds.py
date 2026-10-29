"""Cross-validation folds grouped by well (optionally by a coarser group such as the donor)."""
import json
import os
import numpy as np

def make_folds(well_fractions, k=5, seed=42, groups=None):
    """Partition the wells into `k` test folds; fold i uses fold i+1 (mod k) for validation and the others for training.

    `well_fractions` is {well: ground-truth foreground fraction}: wells are dealt to the folds after sorting by this value,
    so that every fold has a similar organoid density. `groups` is an optional {well: group} (e.g. donor): all the wells of a
    group always end up in the same fold, i.e. no group is shared between training and test data.
    """
    groups = groups or {}
    unit_of = {w: groups.get(w, w) for w in well_fractions}
    members = {}
    for well, unit in unit_of.items():
        members.setdefault(unit, []).append(well)
    if len(members) < 3:
        raise ValueError("At least 3 wells/groups are needed (train, validation and test folds)")
    k = min(k, len(members))
    unit_fraction = {u: float(np.mean([well_fractions[w] for w in ws])) for u, ws in members.items()}
    order = sorted(unit_fraction, key=lambda u: (unit_fraction[u], str(u)))

    rng = np.random.default_rng(seed)
    fold_of = {}
    for start in range(0, len(order), k):
        chunk = order[start:start + k]
        for unit, fold in zip(chunk, rng.permutation(k)[:len(chunk)]):
            fold_of[unit] = int(fold)

    test = [sorted(w for w in well_fractions if fold_of[unit_of[w]] == i) for i in range(k)]
    folds = []
    for i in range(k):
        val_idx = (i + 1) % k
        train = sorted(w for j in range(k) if j not in (i, val_idx) for w in test[j])
        folds.append({"train": train, "val": test[val_idx], "test": test[i]})
    return {"k": k, "seed": seed, "grouped_by": "group" if groups else "well", "groups": {w: str(unit_of[w]) for w in sorted(unit_of)}, "folds": folds}

def check_folds(doc):
    """Sanity checks: every well is tested exactly once, the three sets of a fold are disjoint, groups are never split."""
    folds = doc["folds"]
    tested = [w for f in folds for w in f["test"]]
    assert len(tested) == len(set(tested)), "a well is tested in more than one fold"
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"]) or set(f["val"]) & set(f["test"])), "overlapping sets"
        assert set(f["train"]) | set(f["val"]) | set(f["test"]) == set(tested), "a fold does not cover all the wells"
    if doc.get("grouped_by") == "group":
        group_fold = {}
        for i, f in enumerate(folds):
            for w in f["test"]:
                assert group_fold.setdefault(doc["groups"][w], i) == i, "a group is split across test folds"

def save_folds(doc, path):
    check_folds(doc)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

def load_folds(path):
    with open(path) as f:
        doc = json.load(f)
    check_folds(doc)
    return doc

def fold_wells(doc, fold, role):
    if not 0 <= fold < len(doc["folds"]):
        raise ValueError(f"Fold {fold} out of range (0..{len(doc['folds']) - 1})")
    return doc["folds"][fold][role]
