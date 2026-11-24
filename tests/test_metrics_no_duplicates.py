"""
Reported metrics must not duplicate each other.

Balanced accuracy is, by definition, the macro-averaged recall, and the
support-weighted recall is the overall accuracy. Reporting macro/weighted
recall next to them only repeated the same numbers, so they are not
reported; per-class recall (different information) is kept.
"""

from __future__ import annotations

import importlib.util
import re

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score

from tools.metrics_tracker import ValidationMetricsTracker
from utils import log_format

from .conftest import REPO_ROOT

DUPLICATES = ("recall_macro", "recall_weighted")


def _tracker():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 200)
    preds = np.where(rng.random(200) < 0.7, labels, rng.integers(0, 5, 200))
    t = ValidationMetricsTracker(5, class_names=[f"c{i}" for i in range(5)])
    t.update(1, accuracy_score(labels, preds) * 100, balanced_accuracy_score(labels, preds) * 100, preds, labels)
    return t, labels, preds


def test_duplicates_are_identities():
    """Why the two recall averages are redundant."""
    _, labels, preds = _tracker()
    assert np.isclose(balanced_accuracy_score(labels, preds), recall_score(labels, preds, average="macro"))
    assert np.isclose(accuracy_score(labels, preds), recall_score(labels, preds, average="weighted"))


def test_tracker_results_have_no_duplicate_recall(tmp_path):
    t, _, _ = _tracker()
    metrics = t.get_best_metrics_dict()
    assert not set(DUPLICATES) & set(metrics)
    assert "balanced_acc" in metrics and any(k.startswith("recall_c") for k in metrics)
    csv_path = tmp_path / "cv.csv"
    t.save_cv_results_csv(str(csv_path), "Model")
    header = csv_path.read_text().splitlines()[0].split(",")
    assert not set(DUPLICATES) & set(header)


def test_final_summary_has_no_recall_row(capsys):
    t, _, _ = _tracker()
    log_format.print_final_summary(t, [f"c{i}" for i in range(5)], 1)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)  # drop terminal colour codes
    rows = [line.split()[0] for line in plain.splitlines() if line.strip()]
    assert "Recall" not in rows and "Balanced" in rows


def _load_script(name):
    path = REPO_ROOT / "scripts/reports" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reports_do_not_add_duplicate_columns():
    for name in ("classification_comparison_report", "classification_comparison_split_report"):
        assert not set(DUPLICATES) & set(_load_script(name).DEFAULT_METRICS), name
