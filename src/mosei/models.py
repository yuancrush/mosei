from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}
MODEL_MODALITIES = {
    "text": ("text",),
    "audio": ("audio",),
    "vision": ("vision",),
    "fusion": ("text", "audio", "vision"),
}

EXPERIMENT_MODALITIES = {
    **MODEL_MODALITIES,
    "text_bigru": ("text",),
    "text_audio_bigru": ("text", "audio"),
    "text_audio_gated_bigru": ("text", "audio"),
    "text_vision_bigru": ("text", "vision"),
    "fusion_bigru": ("text", "audio", "vision"),
    "fusion_gated_bigru": ("text", "audio", "vision"),
    "fusion_uncertainty_bigru": ("text", "audio", "vision"),
}


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denominator


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(masked_mean(values, mask))


class MultiTaskBaseline(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if model_name not in MODEL_MODALITIES:
            raise ValueError(f"Unknown model name: {model_name}")
        self.model_name = model_name
        self.modalities = MODEL_MODALITIES[model_name]
        self.projectors = nn.ModuleDict(
            {
                modality: ModalityProjector(
                    INPUT_DIMS[modality], hidden_dim, dropout
                )
                for modality in self.modalities
            }
        )
        fused_dim = hidden_dim * len(self.modalities)
        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embeddings = [
            self.projectors[modality](
                batch[modality], batch[f"{modality}_mask"]
            )
            for modality in self.modalities
        ]
        shared = self.shared(torch.cat(embeddings, dim=-1))
        return {
            "classification_logits": self.classifier(shared),
            "regression": self.regressor(shared).squeeze(-1),
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def input_modalities(self) -> Iterable[str]:
        return self.modalities


class AttentivePool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(values).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (values * weights).sum(dim=1)


class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for the bidirectional GRU")
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = AttentivePool(hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.input(values).masked_fill(~mask.unsqueeze(-1), 0.0)
        # Local modality failure can create holes inside a sequence.  Counting
        # valid steps would make pack_padded_sequence truncate valid timesteps
        # after the hole, so use the final available position as the extent.
        positions = torch.arange(1, mask.shape[1] + 1, device=mask.device)
        lengths = (mask.to(torch.long) * positions).amax(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths, batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.gru(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=values.shape[1]
        )
        return self.output(self.pool(encoded, mask))


class GatedFusion(nn.Module):
    """Use text as the anchor and learn a sample-specific gate for weak modalities."""

    def __init__(self, modalities: Iterable[str], hidden_dim: int) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities or self.modalities[0] != "text":
            raise ValueError("GatedFusion expects text to be the anchor modality")
        self.gates = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid(),
                )
                for modality in self.modalities[1:]
            }
        )
        self.output = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        anchor = embeddings["text"]
        fused = anchor
        for modality in self.modalities[1:]:
            gate = self.gates[modality](torch.cat([anchor, embeddings[modality]], dim=-1))
            fused = fused + gate * embeddings[modality]
        return self.output(fused)


class UncertaintyAwareFusion(nn.Module):
    """Text-anchored fusion using source coverage and Dirichlet vacuity.

    This is a deliberately small, testable adaptation of DCEF rather than a
    reproduction of HyperDiff. Coverage represents feature-source uncertainty;
    vacuity derived from per-modality evidence represents discriminative
    uncertainty. Audio/vision enter as gated residuals, so text remains the
    stable anchor for Attachment 3.
    """

    def __init__(self, modalities: Iterable[str], hidden_dim: int) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.evidence_heads = nn.ModuleDict(
            {modality: nn.Linear(hidden_dim, 3) for modality in self.modalities}
        )
        self.residual_gates = nn.ModuleDict(
            {
                modality: nn.Linear(hidden_dim * 2, 1)
                for modality in self.modalities
                if modality != "text"
            }
        )
        self.output = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        logits = {
            modality: self.evidence_heads[modality](embeddings[modality])
            for modality in self.modalities
        }
        certainties = {}
        coverages = {}
        text_extent = batch["text_mask"].sum(dim=1).clamp_min(1).to(torch.float32)
        for modality in self.modalities:
            evidence = torch.nn.functional.softplus(logits[modality])
            alpha = evidence + 1.0
            certainties[modality] = 1.0 - 3.0 / alpha.sum(dim=-1)
            coverages[modality] = (
                batch[f"{modality}_mask"].sum(dim=1).to(torch.float32) / text_extent
            ).clamp(0.0, 1.0)

        anchor = embeddings["text"]
        fused = anchor
        weights = []
        for modality in self.modalities:
            if modality == "text":
                weights.append(torch.ones_like(certainties[modality]))
                continue
            learned_gate = torch.sigmoid(
                self.residual_gates[modality](
                    torch.cat([anchor, embeddings[modality]], dim=-1)
                ).squeeze(-1)
            )
            weight = coverages[modality] * certainties[modality] * learned_gate
            fused = fused + weight.unsqueeze(-1) * embeddings[modality]
            weights.append(weight)
        return (
            self.output(fused),
            logits,
            torch.stack(weights, dim=-1),
            torch.stack([1.0 - certainties[m] for m in self.modalities], dim=-1),
        )


class MultiTaskTemporal(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if model_name not in EXPERIMENT_MODALITIES or "bigru" not in model_name:
            raise ValueError(f"Unknown temporal model name: {model_name}")
        self.model_name = model_name
        self.modalities = EXPERIMENT_MODALITIES[model_name]
        self.encoders = nn.ModuleDict(
            {
                modality: BiGRUEncoder(INPUT_DIMS[modality], hidden_dim, dropout)
                for modality in self.modalities
            }
        )
        self.gated = "gated_bigru" in model_name
        self.uncertainty_aware = "uncertainty_bigru" in model_name
        if self.uncertainty_aware:
            self.fusion = UncertaintyAwareFusion(self.modalities, hidden_dim)
            fused_dim = hidden_dim
        elif self.gated:
            self.fusion = GatedFusion(self.modalities, hidden_dim)
            fused_dim = hidden_dim
        else:
            self.fusion = None
            fused_dim = hidden_dim * len(self.modalities)
        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embeddings = {
            modality: self.encoders[modality](
                batch[modality], batch[f"{modality}_mask"]
            )
            for modality in self.modalities
        }
        extra_outputs = {}
        if self.uncertainty_aware:
            fused, modality_logits, fusion_weights, modality_uncertainty = self.fusion(
                embeddings, batch
            )
            extra_outputs = {
                "modality_logits": modality_logits,
                "fusion_weights": fusion_weights,
                "modality_uncertainty": modality_uncertainty,
            }
        elif self.fusion is None:
            fused = torch.cat(list(embeddings.values()), dim=-1)
        else:
            fused = self.fusion(embeddings)
        shared = self.shared(fused)
        return {
            "classification_logits": self.classifier(shared),
            "regression": self.regressor(shared).squeeze(-1),
            **extra_outputs,
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def input_modalities(self) -> Iterable[str]:
        return self.modalities


def build_model(
    model_name: str,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name in MODEL_MODALITIES:
        return MultiTaskBaseline(model_name, hidden_dim=hidden_dim, dropout=dropout)
    return MultiTaskTemporal(model_name, hidden_dim=hidden_dim, dropout=dropout)
