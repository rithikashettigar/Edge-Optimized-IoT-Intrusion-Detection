import numpy as np

from edge_ids.evaluation.metrics import classification_metrics


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 0, 1])
    m = classification_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0


def test_false_positive_rate_is_computed_from_benign_flows():
    # Two benign flows, both flagged: in this system that is two legitimate
    # devices cut off the network.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    m = classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["fpr"] == 1.0
    assert m["fp"] == 2
    assert m["tn"] == 0


def test_an_all_benign_predictor_scores_well_on_accuracy_and_zero_on_f1():
    # The exact failure mode that makes accuracy the wrong headline metric.
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.zeros(100, dtype=int)
    m = classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["accuracy"] == 0.95
    assert m["f1"] == 0.0
    assert m["recall"] == 0.0


def test_confusion_counts_sum_to_sample_count():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    m = classification_metrics(y_true, y_pred, y_pred.astype(float))
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 50


def test_single_class_input_does_not_crash_roc_auc():
    y = np.zeros(6, dtype=int)
    m = classification_metrics(y, y, np.zeros(6))
    assert np.isnan(m["roc_auc"]) or 0.0 <= m["roc_auc"] <= 1.0


def test_roc_auc_rewards_a_ranking_better_than_chance():
    y_true = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    assert classification_metrics(y_true, (good > 0.5).astype(int), good)["roc_auc"] == 1.0
