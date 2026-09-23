from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mosei.data import MoseiDataModule, find_aligned_pickle
from mosei.metrics import compute_metrics
from mosei.models import EXPERIMENT_MODALITIES, MODEL_MODALITIES, build_model
from mosei.utils import move_batch, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train four CMU-MOSEI baselines")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(EXPERIMENT_MODALITIES),
        default=list(MODEL_MODALITIES),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-name", default="baselines_seed2026")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--regression-loss",
        choices=("smooth_l1", "mse"),
        default="smooth_l1",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="0 disables class reweighting, 0.5 uses sqrt inverse frequency",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Do not evaluate the test split during iterative model development",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two epochs and a few batches for end-to-end validation",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def batch_losses(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    classification_loss: nn.Module,
    regression_loss: nn.Module,
    regression_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    outputs = model(batch)
    cls_loss = classification_loss(
        outputs["classification_logits"], batch["classification_label"]
    )
    reg_loss = regression_loss(outputs["regression"], batch["regression_label"])
    total = cls_loss + regression_weight * reg_loss
    return total, cls_loss, reg_loss, outputs


def train_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    classification_loss: nn.Module,
    regression_loss: nn.Module,
    regression_weight: float,
    device: torch.device,
    max_batches: int | None,
) -> Dict[str, float]:
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        total, cls_loss, reg_loss, _ = batch_losses(
            model,
            batch,
            classification_loss,
            regression_loss,
            regression_weight,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = batch["classification_label"].shape[0]
        totals += np.array(
            [total.item(), cls_loss.item(), reg_loss.item()], dtype=np.float64
        ) * batch_size
        samples += batch_size
    return {
        "loss": float(totals[0] / samples),
        "classification_loss": float(totals[1] / samples),
        "regression_loss": float(totals[2] / samples),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable[Dict[str, object]],
    classification_loss: nn.Module,
    regression_loss: nn.Module,
    regression_weight: float,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, object]:
    model.eval()
    true_cls: List[np.ndarray] = []
    pred_cls: List[np.ndarray] = []
    true_reg: List[np.ndarray] = []
    pred_reg: List[np.ndarray] = []
    ids: List[str] = []
    totals = np.zeros(3, dtype=np.float64)
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        ids.extend(batch["id"])
        batch = move_batch(batch, device)
        total, cls_loss, reg_loss, outputs = batch_losses(
            model,
            batch,
            classification_loss,
            regression_loss,
            regression_weight,
        )
        batch_size = batch["classification_label"].shape[0]
        totals += np.array(
            [total.item(), cls_loss.item(), reg_loss.item()], dtype=np.float64
        ) * batch_size
        samples += batch_size
        true_cls.append(batch["classification_label"].cpu().numpy())
        pred_cls.append(outputs["classification_logits"].argmax(dim=-1).cpu().numpy())
        true_reg.append(batch["regression_label"].cpu().numpy())
        pred_reg.append(outputs["regression"].clamp(-3.0, 3.0).cpu().numpy())

    y_cls = np.concatenate(true_cls)
    p_cls = np.concatenate(pred_cls)
    y_reg = np.concatenate(true_reg)
    p_reg = np.concatenate(pred_reg)
    result = compute_metrics(y_cls, p_cls, y_reg, p_reg)
    result.update(
        {
            "loss": float(totals[0] / samples),
            "classification_loss": float(totals[1] / samples),
            "regression_loss": float(totals[2] / samples),
            "ids": ids,
            "classification_true": y_cls.tolist(),
            "classification_pred": p_cls.tolist(),
            "regression_true": y_reg.tolist(),
            "regression_pred": p_reg.tolist(),
        }
    )
    return result


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    keys = (
        "loss",
        "classification_loss",
        "regression_loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mae",
        "pearson",
        "selection_score",
    )
    return {key: float(metrics[key]) for key in keys}


def save_confusion_matrix(metrics: Dict[str, object], path: Path, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Neutral", "Positive"],
    )
    display.plot(cmap="Blues", values_format="d", colorbar=False)
    display.ax_.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_history(history: List[Dict[str, float]], path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["valid_macro_f1"] for row in history], label="Macro-F1")
    axes[1].plot(epochs, [row["valid_pearson"] for row in history], label="Pearson")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_model(
    model_name: str,
    module: MoseiDataModule,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(args.seed)
    model = build_model(
        model_name=model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    class_weights = module.class_weights(args.class_weight_power).to(device)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    if args.regression_loss == "mse":
        regression_loss = nn.MSELoss()
    else:
        regression_loss = nn.SmoothL1Loss(beta=0.5)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(model_dir / "tensorboard")
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    epochs = 2 if args.smoke else args.epochs
    max_batches = 4 if args.smoke else None

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model,
            module.loader("train"),
            optimizer,
            classification_loss,
            regression_loss,
            args.regression_weight,
            device,
            max_batches,
        )
        valid_metrics = evaluate(
            model,
            module.loader("valid", shuffle=False),
            classification_loss,
            regression_loss,
            args.regression_weight,
            device,
            max_batches,
        )
        scheduler.step(valid_metrics["selection_score"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "valid_loss": float(valid_metrics["loss"]),
            "valid_macro_f1": float(valid_metrics["macro_f1"]),
            "valid_pearson": float(valid_metrics["pearson"]),
            "valid_mae": float(valid_metrics["mae"]),
            "selection_score": float(valid_metrics["selection_score"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        print(
            f"[{model_name}] epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} valid_loss={row['valid_loss']:.4f} "
            f"macro_f1={row['valid_macro_f1']:.4f} "
            f"mae={row['valid_mae']:.4f} pearson={row['valid_pearson']:.4f}"
        )
        if row["selection_score"] > best_score + 1e-6:
            best_score = row["selection_score"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_valid_metrics": compact_metrics(valid_metrics),
                },
                model_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if not args.smoke and stale_epochs >= args.patience:
            break

    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    valid_metrics = evaluate(
        model,
        module.loader("valid", shuffle=False),
        classification_loss,
        regression_loss,
        args.regression_weight,
        device,
        max_batches if args.smoke else None,
    )
    test_metrics = None
    if not args.validation_only:
        test_metrics = evaluate(
            model,
            module.loader("test", shuffle=False),
            classification_loss,
            regression_loss,
            args.regression_weight,
            device,
            max_batches if args.smoke else None,
        )
    elapsed = time.perf_counter() - started
    writer.close()

    write_json(model_dir / "history.json", history)
    write_json(model_dir / "valid_predictions.json", valid_metrics)
    if test_metrics is not None:
        write_json(model_dir / "test_predictions.json", test_metrics)
        save_confusion_matrix(
            test_metrics, model_dir / "test_confusion_matrix.png", f"{model_name} test"
        )
    save_history(history, model_dir / "training_history.png", model_name)
    result = {
        "model": model_name,
        "modalities": list(model.modalities),
        "parameters": model.trainable_parameter_count(),
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "valid": compact_metrics(valid_metrics),
    }
    if test_metrics is not None:
        result["test"] = compact_metrics(test_metrics)
    write_json(model_dir / "summary.json", result)
    return result


def save_summary_csv(results: List[Dict[str, object]], path: Path) -> None:
    fields = [
        "model",
        "modalities",
        "parameters",
        "best_epoch",
        "elapsed_seconds",
        "split",
        "loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mae",
        "pearson",
        "selection_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for split in ("valid", "test"):
                if split not in result:
                    continue
                metrics = result[split]
                writer.writerow(
                    {
                        "model": result["model"],
                        "modalities": "+".join(result["modalities"]),
                        "parameters": result["parameters"],
                        "best_epoch": result["best_epoch"],
                        "elapsed_seconds": round(float(result["elapsed_seconds"]), 3),
                        "split": split,
                        **{key: metrics[key] for key in fields if key in metrics},
                    }
                )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_path = args.data or find_aligned_pickle(PROJECT_ROOT)
    suffix = "_smoke" if args.smoke else ""
    run_dir = args.output_dir / f"{args.run_name}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data: {data_path.resolve()}")
    print(f"Device: {device}")
    module = MoseiDataModule(
        data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "data": str(data_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device": str(device),
            "split_sizes": module.split_sizes(),
            "class_weights": module.class_weights(args.class_weight_power).tolist(),
            "feature_stats": {
                modality: stats.to_dict()
                for modality, stats in module.stats.items()
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )
    results = [
        train_model(model_name, module, args, run_dir, device)
        for model_name in args.models
    ]
    write_json(run_dir / "summary.json", results)
    save_summary_csv(results, run_dir / "summary.csv")
    print(f"Finished. Summary: {(run_dir / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
