"""One explicit classification threshold, and the bag-level training loss."""

import pytest
import torch
import torch.nn as nn

from conftest import ROOT, make_cfg
from lightning_modules.mil_classifier import MILClassifier
from models import get_model

DIM = 8


def build(tmp_path, **lit_overrides):
    cfg = make_cfg(tmp_path / "m.parquet", DIM, tmp_path, **lit_overrides)
    model = get_model(cfg.model.name, **cfg.model.params)
    return MILClassifier(model, cfg), model


def synthetic_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 6, DIM, generator=g)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    labels = torch.tensor([0, 1, 1])
    return x, labels, mask, ["a", "b", "c"], [[None] * 6] * 3


# ------------------------------------------------------------------ threshold

def test_no_threshold_literal_in_lightning_modules():
    hits = []
    for path in (ROOT / "lightning_modules").glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "0.5" in line:
                hits.append(f"{path.name}:{number}: {line.strip()}")
    assert hits == []


def test_threshold_is_required(tmp_path):
    cfg = make_cfg(tmp_path / "m.parquet", DIM, tmp_path)
    del cfg.lit_module["threshold"]
    model = get_model(cfg.model.name, **cfg.model.params)
    with pytest.raises(ValueError, match="lit_module.threshold"):
        MILClassifier(model, cfg)


@pytest.mark.parametrize("threshold", [0.3, 0.7])
def test_every_metric_and_the_predictions_use_the_configured_threshold(tmp_path, threshold):
    lit, _ = build(tmp_path, threshold=threshold)
    for name in ["train_acc", "val_acc", "val_f1", "val_recall", "val_precision",
                 "test_acc", "test_f1", "test_recall", "test_precision"]:
        assert getattr(lit, name).threshold == threshold, name
    probs = torch.tensor([0.1, 0.4, 0.6, 0.9])
    expected = (probs > threshold).long()
    assert torch.equal(lit._binarize(probs), expected)


def test_predict_step_uses_the_configured_threshold(tmp_path):
    lit, _ = build(tmp_path, threshold=0.99)
    lit.eval()
    x, labels, mask, ids, patch_ids = synthetic_batch()
    preds, probs, *_ = lit.predict_step((x, labels, mask, ids, patch_ids), 0)
    assert torch.equal(preds, (probs > 0.99).long())


# ------------------------------------------------------------------ loss

def reference_bag_loss(lit, model, batch):
    x, labels, mask, *_ = batch
    logits = model(x, mask)[0].view(-1)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([lit.cfg.lit_module.pos_weight]))(logits, labels.float())


def test_training_loss_is_exactly_the_pos_weighted_bce(tmp_path):
    lit, model = build(tmp_path)
    lit.train()
    batch = synthetic_batch()
    loss, *_ = lit._shared_step(batch, 0)
    assert torch.equal(loss, reference_bag_loss(lit, model, batch))


def test_instance_classifiers_receive_no_gradient(tmp_path):
    lit, model = build(tmp_path)
    lit.train()
    loss, *_ = lit._shared_step(synthetic_batch(), 0)
    loss.backward()
    assert all(p.grad is None for p in model.model.inst_classifiers.parameters())
