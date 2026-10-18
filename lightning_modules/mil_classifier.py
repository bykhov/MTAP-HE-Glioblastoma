import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR

# Schema of fold{k}_test_predictions.csv and fold{k}_val_predictions.csv
PREDICTION_COLUMNS = ["patient_id", "true_label", "prediction", "probability", "logit", "fold"]

ATTENTION_MODELS = ["clam_sb"]


class MILClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for MIL classification.

    Args:
        model: the MIL model (returns ``logits, model_specific_output, attention_scores``).
        cfg: the full config. ``lit_module.threshold`` is required: it is the one classification
            threshold used for every metric and for the saved predictions.
        fold_index: index of the cross-validation run; used in the output file names and the
            ``fold`` column of the prediction CSVs.
        output_dir: directory of the experiment. When given, ``fold{k}_test_predictions.csv`` and
            ``fold{k}_val_predictions.csv`` are written there and the per-fold artifacts (attention
            outputs) go to ``output_dir/fold{k}/``. When None nothing is written to disk.
    """

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], fold_index: Optional[int] = None,
                 output_dir: Optional[os.PathLike] = None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.fold_index = fold_index
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None and fold_index is None:
            raise ValueError("fold_index is required when output_dir is given.")
        self.save_hyperparameters(ignore=['model', 'output_dir'])

        # The single classification threshold
        threshold = self.cfg.lit_module.get('threshold')
        if threshold is None:
            raise ValueError("lit_module.threshold must be set in the config.")
        self.threshold = float(threshold)

        # Bag-level loss, weighted for the positive class
        pos_weight = self.cfg.lit_module.get('pos_weight', 1.0)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics. The threshold is given to every metric that has one; predictions are binarized
        # with the same threshold in _binarize() before they are passed in.
        self.train_acc = torchmetrics.Accuracy(task="binary", threshold=self.threshold)
        self.val_acc = torchmetrics.Accuracy(task="binary", threshold=self.threshold)
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary", threshold=self.threshold)
        self.val_recall = torchmetrics.Recall(task="binary", threshold=self.threshold)
        self.val_precision = torchmetrics.Precision(task="binary", threshold=self.threshold)

        # Test metrics
        self.test_acc = torchmetrics.Accuracy(task="binary", threshold=self.threshold)
        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary", threshold=self.threshold)
        self.test_recall = torchmetrics.Recall(task="binary", threshold=self.threshold)
        self.test_precision = torchmetrics.Precision(task="binary", threshold=self.threshold)

        # Store validation and test outputs for writing the prediction files at the end of the epoch
        self.validation_outputs = []
        self.test_outputs = []
        self.best_val_auroc = -math.inf

    def _binarize(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs > self.threshold).long()

    @property
    def fold_dir(self) -> Optional[Path]:
        return self.output_dir / f"fold{self.fold_index}" if self.output_dir is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x, mask)

    def _shared_step(self, batch, batch_idx):
        instances, labels, mask, bag_ids, patch_ids = batch
        if instances.numel() == 0:  # Handle empty batch from collate_fn
            return None, None, None, 0, None, None, None, None

        # --- Apply Patch Dropout during training ---
        patch_dropout_rate = self.cfg.lit_module.get('patch_dropout_rate', 0.0)
        small_bag_threshold = self.cfg.lit_module.get('patch_dropout_small_bag_threshold', 0)
        if self.training and patch_dropout_rate > 0:
            # Get the number of real patches in each bag
            num_real_patches = mask.sum(dim=1)
            # Create a dropout mask for the patches that are actually present
            dropout_mask = torch.rand_like(mask.float()) > patch_dropout_rate
            # Only apply dropout to bags larger than the threshold
            dropout_mask[num_real_patches <= small_bag_threshold] = True
            # Combine with the original mask to ensure we don't "undrop" padded patches
            mask = mask & dropout_mask

        logits, model_specific_output, attention_scores = self.forward(instances, mask)
        logits = logits.view(-1)
        loss = self.loss_fn(logits, labels.float())

        batch_size = labels.size(0)
        return loss, logits, labels, batch_size, model_specific_output, bag_ids, patch_ids, attention_scores

    def training_step(self, batch, batch_idx):
        loss, logits, labels, batch_size, _, _, _, _ = self._shared_step(batch, batch_idx)
        if loss is None: return None

        probs = torch.sigmoid(logits)
        preds = self._binarize(probs)
        self.train_acc(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels, batch_size, model_specific_output, bag_ids, patch_ids, attention_scores = self._shared_step(
            batch, batch_idx)
        if loss is None: return None

        probs = torch.sigmoid(logits)
        preds = self._binarize(probs)

        self.validation_outputs.append({
            'bag_ids': bag_ids,
            'labels': labels.cpu(),
            'preds': preds.cpu(),
            'probs': probs.cpu(),
            'logits': logits.cpu()
        })

        self.val_acc(preds, labels)
        self.val_auroc(probs, labels)
        self.val_f1(preds, labels)
        self.val_recall(preds, labels)
        self.val_precision(preds, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/auroc", self.val_auroc, on_epoch=True, batch_size=batch_size)
        self.log("val/f1", self.val_f1, on_epoch=True, batch_size=batch_size)
        self.log("val/recall", self.val_recall, on_epoch=True, batch_size=batch_size)
        self.log("val/precision", self.val_precision, on_epoch=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        # Keep the validation predictions of the best epoch on disk. A new best is a strict
        # improvement of val/auroc, which is exactly the rule ModelCheckpoint(save_top_k=1) uses,
        # so the file always belongs to the checkpoint that is selected for testing.
        try:
            if (not self.validation_outputs or self.trainer.sanity_checking
                    or not self.trainer.is_global_zero):
                return
            current_auroc = float(self.val_auroc.compute())
            if not current_auroc > self.best_val_auroc:
                return
            self.best_val_auroc = current_auroc
            self._write_predictions(self.validation_outputs, "val")
        finally:
            self.validation_outputs.clear()

    def _predictions_frame(self, outputs) -> pd.DataFrame:
        return pd.DataFrame({
            "patient_id": [pid for output in outputs for pid in output['bag_ids']],
            "true_label": torch.cat([output['labels'] for output in outputs]).numpy(),
            "prediction": torch.cat([output['preds'] for output in outputs]).numpy(),
            "probability": torch.cat([output['probs'] for output in outputs]).float().numpy(),
            "logit": torch.cat([output['logits'] for output in outputs]).float().numpy(),
            "fold": self.fold_index,
        })[PREDICTION_COLUMNS]

    def _write_predictions(self, outputs, split: str) -> pd.DataFrame:
        results_df = self._predictions_frame(outputs)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(self.output_dir / f"fold{self.fold_index}_{split}_predictions.csv", index=False)
        return results_df

    def test_step(self, batch, batch_idx):
        loss, logits, labels, batch_size, model_specific_output, bag_ids, patch_ids, attention_scores = self._shared_step(batch, batch_idx)
        if loss is None: return None

        probs = torch.sigmoid(logits)
        preds = self._binarize(probs)

        self.test_acc(preds, labels)
        self.test_auroc(probs, labels)
        self.test_f1(preds, labels)
        self.test_recall(preds, labels)
        self.test_precision(preds, labels)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/acc", self.test_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/auroc", self.test_auroc, on_epoch=True, batch_size=batch_size)
        self.log("test/f1", self.test_f1, on_epoch=True, batch_size=batch_size)
        self.log("test/recall", self.test_recall, on_epoch=True, batch_size=batch_size)
        self.log("test/precision", self.test_precision, on_epoch=True, batch_size=batch_size)

        output_data = {
            'bag_ids': bag_ids, # This is patient_id
            'patch_ids': patch_ids,
            'labels': labels.cpu(),
            'preds': preds.cpu(),
            'probs': probs.cpu(),
            'logits': logits.cpu()
        }

        # --- Attention Analysis and Logging ---
        if attention_scores is not None and self.cfg.model.name in ATTENTION_MODELS:
            _, _, mask, _, _ = batch
            mask = mask.bool()
            num_real_patches = mask.sum(dim=1)

            if attention_scores.ndim == 3:
                attention_scores = attention_scores.squeeze(-1)

            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)
            normalized_attention = torch.softmax(attention_scores, dim=1)
            masked_normalized_attention = normalized_attention * mask

            max_attn_per_bag = torch.max(masked_normalized_attention, dim=1).values
            mean_attn_per_bag = torch.sum(masked_normalized_attention, dim=1) / num_real_patches.clamp(min=1)

            ipr = 1.0 / torch.sum(masked_normalized_attention ** 2, dim=1)
            ipr[num_real_patches == 0] = 0.0

            output_data.update({
                'max_attn_weights': max_attn_per_bag.cpu(),
                'mean_attn_weights': mean_attn_per_bag.cpu(),
                'ipr_score': ipr.cpu(),
                'attention_scores': masked_normalized_attention.cpu(),
                'mask': mask.cpu()
            })

        self.test_outputs.append(output_data)
        return {"loss": loss, "labels": labels, "preds": preds}

    def on_test_epoch_end(self):
        if not self.test_outputs or not self.trainer.is_global_zero:
            return

        # --- Test predictions: fold{k}_test_predictions.csv ---
        self._write_predictions(self.test_outputs, "test")

        # --- Attention outputs, written under output_dir/fold{k}/ ---
        if self.cfg.model.name in ATTENTION_MODELS and self.output_dir is not None:
            self._write_attention_outputs()

        self.test_outputs.clear()

    def _write_attention_outputs(self):
        """
        Per-patient attention CSVs (``attention_maps/<patient_id>.csv``: slide_id, patch_id,
        attention_score) and the aggregate table ``attention_analysis.csv``.
        """
        attention_dir = self.fold_dir / "attention_maps"
        attention_dir.mkdir(parents=True, exist_ok=True)
        table_data = []

        for output in self.test_outputs:
            if 'attention_scores' not in output:
                continue

            patient_ids_batch = output['bag_ids']
            patch_ids_batch = output['patch_ids']
            attention_scores_batch = output['attention_scores']
            mask_batch = output['mask']

            for i in range(len(patient_ids_batch)):
                patient_id = patient_ids_batch[i]

                # --- Aggregate table row (one bag per patient) ---
                table_data.append([
                    patient_id,
                    output['labels'][i].item(),
                    output['preds'][i].item(),
                    output['probs'][i].item(),
                    output['max_attn_weights'][i].item(),
                    output['mean_attn_weights'][i].item(),
                    output['ipr_score'][i].item()
                ])

                # --- Per-patient CSV ---
                scores_for_bag = attention_scores_batch[i]
                patch_ids_for_bag = patch_ids_batch[i]
                mask_for_bag = mask_batch[i]

                valid_scores = scores_for_bag[mask_for_bag]
                valid_patch_ids = [pid for pid, m in zip(patch_ids_for_bag, mask_for_bag) if m]

                if not valid_patch_ids:
                    continue

                patient_df = pd.DataFrame({
                    'patch_id': valid_patch_ids,
                    'attention_score': valid_scores.numpy()
                })

                patient_df['slide_id'] = patient_df['patch_id'].apply(lambda x: '-'.join(x.split('-')[:-1]))

                patient_df = patient_df[['slide_id', 'patch_id', 'attention_score']]
                patient_df.to_csv(attention_dir / f"{patient_id}.csv", index=False)

        columns = ["patient_id", "label", "prediction", "probability", "max_attn_weight", "mean_attn_weight", "ipr_score"]
        pd.DataFrame(table_data, columns=columns).to_csv(self.fold_dir / "attention_analysis.csv", index=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        instances, _, mask, bag_ids, patch_ids = batch
        logits, model_specific_output, attention_scores = self.forward(instances, mask)
        logits = logits.view(-1)
        probs = torch.sigmoid(logits)
        preds = self._binarize(probs)
        return preds, probs, model_specific_output, bag_ids, patch_ids, attention_scores

    def configure_optimizers(self):
        opt_cfg = self.cfg['lit_module']['optimizer']
        optimizer_name = opt_cfg['name']
        optimizer_params = {k: v for k, v in opt_cfg.items() if k != 'name'}

        optimizer_class = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        scheduler_cfg = self.cfg['lit_module'].get('scheduler')
        if not scheduler_cfg or not scheduler_cfg.get('name'):
            return optimizer

        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg['name'])
        main_scheduler = scheduler_class(optimizer, **scheduler_cfg.get('params', {}))

        warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            def warmup_lr(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)

            # The main scheduler will start after the warmup
            sequential_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sequential_scheduler,
                    "monitor": "val/loss",
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": main_scheduler,
                "monitor": "val/loss",
            },
        }
