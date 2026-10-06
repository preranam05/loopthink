import csv
import json
import os

import numpy as np
import pytest

from loopthink import custom


def write_csv(path, rows, header=("text", "intent")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    return str(path)


def rows_for(n_intents=3, per=12):
    return [(f"request {i} about topic {k}", f"intent_{k}") for k in range(n_intents) for i in range(per)]


def test_reads_cleans_and_reports(tmp_path):
    rows = rows_for() + [("  request 0 about   topic 0 ", "intent_0"), ("", "intent_1"), ("orphan", ""),
                         ("request 1 about topic 0", "intent_2"), ("Mixed Case", "Reset Password!")]
    pairs, notes = custom.read_examples(write_csv(tmp_path / "d.csv", rows, ("Request", "Label")))
    assert len(pairs) == 37 and ("Mixed Case", "reset_password") in pairs          # label normalised
    joined = " ".join(notes)
    assert "2 rows with an empty" in joined and "2 duplicate" in joined and "two different intents" in joined


def test_rejects_data_that_cannot_train(tmp_path):
    with pytest.raises(custom.DataError, match="at least 2 intents"):
        custom.check([("a", "x")] * 20)
    with pytest.raises(custom.DataError, match="intent_1 \\(3\\)"):
        custom.check(rows_for(1, 12) + [(f"q{i}", "intent_1") for i in range(3)])
    with pytest.raises(custom.DataError, match="no intent column"):
        custom.read_examples(write_csv(tmp_path / "bad.csv", [("a", "b")], ("text", "owner")))
    warn = custom.check(rows_for(2, 12) + [(f"big {i}", "intent_9") for i in range(200)])
    assert any("fewer than 30" in w for w in warn) and any("uneven" in w for w in warn)


def test_split_is_stratified_and_disjoint():
    pairs = rows_for(4, 25)
    s = custom.split_examples(pairs, seed=1)
    texts = [t for part in s.values() for t, _ in part]
    assert len(texts) == len(set(texts)) == 100                                     # nothing lost, nothing shared
    for part, n in (("val", 4), ("test", 4), ("train", 17)):
        counts = {}
        for _, l in s[part]:
            counts[l] = counts.get(l, 0) + 1
        assert set(counts.values()) == {n}
    assert custom.split_examples(pairs, seed=1) == s and custom.split_examples(pairs, seed=2) != s
    d = custom.to_dataset(s, ["u1", "u2", "u3", "u4", "u5"])
    assert len(d["oos_val"]) == 2 and len(d["oos_test"]) == 3 and d["train"] == s["train"]


def test_unknown_file_with_or_without_header(tmp_path):
    a = tmp_path / "u.csv"; a.write_text("text\nwhere is lunch\nWhere is lunch\nbook a room\n")
    b = tmp_path / "v.csv"; b.write_text("where is lunch\nbook a room\n")
    assert custom.read_texts(str(a)) == custom.read_texts(str(b)) == ["where is lunch", "book a room"]


def test_report_numbers():
    rows, conf = custom.per_intent([0, 0, 1, 1, 1], [0, 1, 1, 1, 1], ["a", "b"])
    assert rows == [("a", 2, 0.5), ("b", 3, 1.0)] and conf == [("a", "b", 1)]
    rt = custom.routing_table([0.99, 0.6, 0.95, 0.3], [1, 0, 1, 0], [("Balanced", 0.9), ("Never", 1.0)])
    assert rt[0]["escalated"] == 0.5 and rt[0]["kept_accuracy"] == 1.0 and rt[1]["kept"] == 0
    lo, hi = custom.bootstrap_ci([1] * 90 + [0] * 10)
    assert 0.8 < lo < 0.9 < hi <= 0.97 and custom.bootstrap_ci([1, 1, 1]) == (1.0, 1.0)


def test_example_dataset_is_usable():
    root = os.path.join(os.path.dirname(__file__), "..", "examples", "helpdesk")
    pairs, notes = custom.read_examples(os.path.join(root, "requests.csv"))
    assert len(pairs) == 200 and not notes and len({l for _, l in pairs}) == 8
    custom.check(pairs)
    unknown = custom.read_texts(os.path.join(root, "unknown.csv"))
    assert len(unknown) == 36 and not {u.lower() for u in unknown} & {t.lower() for t, _ in pairs}
