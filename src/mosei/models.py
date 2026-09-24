from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F


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
    "fusion_task_gated_bigru": ("text", "audio", "vision"),
    "fusion_output_residual_bigru": ("text", "audio", "vision"),
    "text_bert_finetune": ("text",),
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


class TaskSpecificTrustedFusion(nn.Module):
    """文本锚定的分类/回归解耦可信残差融合。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weak_modalities = ("audio", "vision")
        self.residuals = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for modality in self.weak_modalities
            }
        )
        gate_input_dim = hidden_dim * 2 + 3
        self.classification_gates = nn.ModuleDict(
            {
                modality: self._make_gate(gate_input_dim, hidden_dim)
                for modality in self.weak_modalities
            }
        )
        self.regression_gates = nn.ModuleDict(
            {
                modality: self._make_gate(gate_input_dim, hidden_dim)
                for modality in self.weak_modalities
            }
        )
        self.classification_output = nn.LayerNorm(hidden_dim)
        self.regression_output = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _make_gate(input_dim: int, hidden_dim: int) -> nn.Sequential:
        gate = nn.Sequential(
            nn.Linear(input_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        # 初始时弱模态只产生小残差，避免一开始破坏已验证的文本锚点。
        nn.init.constant_(gate[-1].bias, -3.0)
        return gate

    @staticmethod
    def _availability_features(
        batch: Dict[str, torch.Tensor], modality: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_mask = batch["text_mask"].bool()
        modality_mask = batch[f"{modality}_mask"].bool() & text_mask
        text_length = text_mask.sum(dim=1).clamp_min(1).to(torch.float32)
        coverage = modality_mask.sum(dim=1).to(torch.float32) / text_length

        run = torch.zeros_like(text_length)
        longest = torch.zeros_like(text_length)
        for position in range(text_mask.shape[1]):
            missing = text_mask[:, position] & ~modality_mask[:, position]
            run = torch.where(missing, run + 1.0, torch.zeros_like(run))
            longest = torch.maximum(longest, run)
        longest_gap_ratio = longest / text_length
        descriptors = torch.stack(
            [coverage.clamp(0.0, 1.0), longest_gap_ratio.clamp(0.0, 1.0)], dim=-1
        )
        return descriptors, coverage.clamp(0.0, 1.0)

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = embeddings["text"]
        classification_fused = anchor
        regression_fused = anchor
        weights = []
        for modality in self.weak_modalities:
            availability, coverage = self._availability_features(batch, modality)
            agreement = F.cosine_similarity(
                anchor, embeddings[modality], dim=-1
            ).unsqueeze(-1)
            gate_input = torch.cat(
                [anchor, embeddings[modality], availability, agreement], dim=-1
            )
            classification_weight = coverage * torch.sigmoid(
                self.classification_gates[modality](gate_input).squeeze(-1)
            )
            regression_weight = coverage * torch.sigmoid(
                self.regression_gates[modality](gate_input).squeeze(-1)
            )
            residual = self.residuals[modality](embeddings[modality])
            classification_fused = (
                classification_fused + classification_weight.unsqueeze(-1) * residual
            )
            regression_fused = (
                regression_fused + regression_weight.unsqueeze(-1) * residual
            )
            weights.append(
                torch.stack([classification_weight, regression_weight], dim=-1)
            )
        return (
            self.classification_output(classification_fused),
            self.regression_output(regression_fused),
            torch.stack(weights, dim=1),
        )


class OutputResidualFusion(nn.Module):
    """在强文本预测上学习小幅音视频纠错，初始输出严格等于文本模型。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weak_modalities = ("audio", "vision")
        self.classification_corrections = nn.ModuleDict(
            {modality: nn.Linear(hidden_dim, 3) for modality in self.weak_modalities}
        )
        self.regression_corrections = nn.ModuleDict(
            {modality: nn.Linear(hidden_dim, 1) for modality in self.weak_modalities}
        )
        gate_dim = hidden_dim * 2 + 4
        self.classification_gates = nn.ModuleDict(
            {
                modality: TaskSpecificTrustedFusion._make_gate(gate_dim, hidden_dim)
                for modality in self.weak_modalities
            }
        )
        self.regression_gates = nn.ModuleDict(
            {
                modality: TaskSpecificTrustedFusion._make_gate(gate_dim, hidden_dim)
                for modality in self.weak_modalities
            }
        )
        for head in list(self.classification_corrections.values()) + list(
            self.regression_corrections.values()
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.audio_scale = 1.0
        self.vision_scale = 1.0

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        anchor_logits: torch.Tensor,
        anchor_regression: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = embeddings["text"]
        anchor_probability = torch.softmax(anchor_logits, dim=-1)
        entropy = -(
            anchor_probability * anchor_probability.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / torch.log(
            torch.tensor(3.0, device=anchor.device, dtype=anchor.dtype)
        )
        classification = anchor_logits
        regression = anchor_regression
        weights = []
        for modality in self.weak_modalities:
            availability, coverage = TaskSpecificTrustedFusion._availability_features(
                batch, modality
            )
            agreement = F.cosine_similarity(
                anchor, embeddings[modality], dim=-1
            ).unsqueeze(-1)
            gate_input = torch.cat(
                [anchor, embeddings[modality], availability, agreement, entropy], dim=-1
            )
            classification_weight = coverage * torch.sigmoid(
                self.classification_gates[modality](gate_input).squeeze(-1)
            )
            regression_weight = coverage * torch.sigmoid(
                self.regression_gates[modality](gate_input).squeeze(-1)
            )
            classification = classification + classification_weight.unsqueeze(-1) * (
                self.classification_corrections[modality](embeddings[modality])
                * float(getattr(self, f"{modality}_scale"))
            )
            regression = regression + regression_weight * (
                self.regression_corrections[modality](embeddings[modality]).squeeze(-1)
                * float(getattr(self, f"{modality}_scale"))
            )
            weights.append(
                torch.stack([classification_weight, regression_weight], dim=-1)
            )
        return classification, regression, torch.stack(weights, dim=1)


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
        self.task_gated = model_name == "fusion_task_gated_bigru"
        self.output_residual = model_name == "fusion_output_residual_bigru"
        if self.output_residual:
            self.fusion = OutputResidualFusion(hidden_dim)
            fused_dim = hidden_dim
            self.modality_classifiers = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, 3) for modality in ("audio", "vision")}
            )
            self.modality_regressors = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, 1) for modality in ("audio", "vision")}
            )
        elif self.task_gated:
            self.fusion = TaskSpecificTrustedFusion(hidden_dim)
            fused_dim = hidden_dim
            self.modality_classifiers = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, 3) for modality in ("audio", "vision")}
            )
            self.modality_regressors = nn.ModuleDict(
                {modality: nn.Linear(hidden_dim, 1) for modality in ("audio", "vision")}
            )
        elif self.uncertainty_aware:
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
        if self.output_residual:
            anchor_shared = self.shared(embeddings["text"])
            anchor_logits = self.classifier(anchor_shared)
            anchor_regression = self.regressor(anchor_shared).squeeze(-1)
            classification, regression, task_fusion_weights = self.fusion(
                embeddings, batch, anchor_logits, anchor_regression
            )
            return {
                "classification_logits": classification,
                "regression": regression,
                "anchor_classification_logits": anchor_logits,
                "anchor_regression": anchor_regression,
                "modality_logits": {
                    modality: self.modality_classifiers[modality](embeddings[modality])
                    for modality in ("audio", "vision")
                },
                "modality_regressions": {
                    modality: self.modality_regressors[modality](
                        embeddings[modality]
                    ).squeeze(-1)
                    for modality in ("audio", "vision")
                },
                "task_fusion_weights": task_fusion_weights,
            }
        if self.task_gated:
            classification_fused, regression_fused, task_fusion_weights = self.fusion(
                embeddings, batch
            )
            classification_shared = self.shared(classification_fused)
            regression_shared = self.shared(regression_fused)
            anchor_shared = self.shared(embeddings["text"])
            return {
                "classification_logits": self.classifier(classification_shared),
                "regression": self.regressor(regression_shared).squeeze(-1),
                "anchor_classification_logits": self.classifier(anchor_shared),
                "anchor_regression": self.regressor(anchor_shared).squeeze(-1),
                "modality_logits": {
                    modality: self.modality_classifiers[modality](embeddings[modality])
                    for modality in ("audio", "vision")
                },
                "modality_regressions": {
                    modality: self.modality_regressors[modality](
                        embeddings[modality]
                    ).squeeze(-1)
                    for modality in ("audio", "vision")
                },
                "task_fusion_weights": task_fusion_weights,
            }
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


class BertTopLayerMultiTask(nn.Module):
    """只微调通用BERT顶部两层的文本多任务模型。"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name = "text_bert_finetune"
        self.modalities = ("text",)
        self.bert = AutoModel.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        for parameter in self.bert.parameters():
            parameter.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        self.pool = AttentivePool(self.bert.config.hidden_size)
        self.shared = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self.bert(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_mask"].to(torch.long),
            token_type_ids=batch["text_token_type_ids"],
        ).last_hidden_state
        shared = self.shared(self.pool(encoded, batch["text_mask"]))
        return {
            "classification_logits": self.classifier(shared),
            "regression": self.regressor(shared).squeeze(-1),
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
    if model_name == "text_bert_finetune":
        return BertTopLayerMultiTask(hidden_dim=hidden_dim, dropout=dropout)
    if model_name in MODEL_MODALITIES:
        return MultiTaskBaseline(model_name, hidden_dim=hidden_dim, dropout=dropout)
    return MultiTaskTemporal(model_name, hidden_dim=hidden_dim, dropout=dropout)
