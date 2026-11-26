"""Folds (train/val/test wells) grouped by well, e.g. the leave-one-dataset-out folds written by hynet-prepare-pool."""
import json
import os

def check_folds(doc):
    """Sanity checks: every well is tested exactly once, the three sets of a fold are disjoint, groups are never split.

    A "holdout" file (e.g. leave-one-dataset-out) has folds whose train/val/test sets need not cover the same wells.
    """
    folds = doc["folds"]
    tested = [w for f in folds for w in f["test"]]
    assert len(tested) == len(set(tested)), "a well is tested in more than one fold"
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"]) or set(f["val"]) & set(f["test"])), "overlapping sets"
        if not doc.get("holdout"):
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
