import torch
import torch.nn as nn
from torchmil.models.clam import CLAM_SB
from typing import Tuple, Optional


class CLAMSBModel(nn.Module):
    """
    torchmil CLAM_SB with a LayerNorm on the input embeddings.

    Training uses the bag-level loss only (``MILClassifier``), so the trained network is a gated
    attention MIL model (gated ABMIL): LayerNorm, gated attention pooling and one linear head.
    The instance classifiers that CLAM_SB contains are not trained.
    """

    def __init__(self, input_dim: int, n_classes: int, **kwargs):
        super().__init__()
        if n_classes != 1:
            raise ValueError(
                f"CLAMSBModel wrapper expected n_classes=1 for torchmil.CLAM_SB, but got {n_classes}."
            )

        self.norm = nn.LayerNorm(input_dim)
        self.model = CLAM_SB(in_shape=(input_dim,), **kwargs)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_instances, input_dim)
            mask (torch.Tensor): Boolean mask of shape (batch_size, n_instances) indicating valid instances.
        """
        x_norm = self.norm(x)

        bag_logits, attention_scores = self.model(x_norm, mask=mask, return_att=True)
        bag_logits = bag_logits.unsqueeze(1)

        return bag_logits, None, attention_scores
