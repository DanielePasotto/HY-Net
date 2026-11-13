import pytest

from hynet import data, folds
from hynet.cli.prepare_pool import DOMAINS, lodo_folds

def item(domain, role, stem):
    return {"domain": domain, "role": role, "name": f"{domain}-{stem}_Interval0.png"}

def pool_items():
    items = []
    for domain, n_train, n_val, n_test in (("ours", 6, 2, 2), ("orgaseg", 4, 1, 1), ("organoid", 10, 1, 3)):
        for role, n in (("train", n_train), ("val", n_val), ("test", n_test)):
            items += [item(domain, role, f"{role}{i}") for i in range(n)]
    return items

def test_domain_of_and_weights():
    assert data.domain_of("/x/IMG_orgaseg-0002_1141294_abc_Interval0.png") == "orgaseg"
    assert data.domain_of("IMG_organoid-mouse-b_image0007_Interval0.png") == "organoid"
    assert data.domain_of("IMG_ours-B11_Interval1.tif") == "ours"
    assert data.domain_of("IMG_B11_Interval1.tif") == ""
    paths = ["IMG_ours-A_Interval0.png"] * 2 + ["IMG_organoid-B_Interval0.png"] * 8
    weights = data.domain_weights(paths)
    assert sum(weights[:2]) == pytest.approx(sum(weights[2:])) == pytest.approx(1.0)

def test_lodo_folds_train_on_the_other_domains():
    docs = lodo_folds(pool_items())
    assert set(docs) == set(DOMAINS)
    for held_out, doc in docs.items():
        fold = doc["folds"][0]
        assert doc["holdout"] and not any(w.startswith(held_out) for w in fold["train"] + fold["val"])
        assert all(w.startswith(held_out) for w in fold["test"])
        assert {w.split("-")[0] for w in fold["train"]} == set(DOMAINS) - {held_out}
        folds.check_folds(doc)
    assert len(docs["ours"]["folds"][0]["test"]) == 10      # every image of ours
    assert len(docs["organoid"]["folds"][0]["test"]) == 3   # only the official testing images
    assert len(docs["organoid"]["folds"][0]["train"]) == 6 + 4

def test_holdout_folds_may_leave_wells_out_but_not_overlap():
    doc = {"k": 1, "seed": 0, "grouped_by": "well", "holdout": True, "groups": {}, "folds": [{"train": ["a"], "val": ["b"], "test": ["c"]}]}
    folds.check_folds(doc)
    doc["folds"][0]["val"] = ["a"]
    with pytest.raises(AssertionError):
        folds.check_folds(doc)
    doc.pop("holdout")
    doc["folds"][0]["val"] = ["b"]
    with pytest.raises(AssertionError):
        folds.check_folds(doc)
