from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METRICS = ["accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "selection_score"]


def load_row(path: Path, model: str, seed: int) -> dict:
    frame = pd.read_csv(path)
    row = frame.loc[(frame["model"] == model) & (frame["split"] == "valid")].iloc[0]
    return {"seed": seed, "model": model, **{metric: row[metric] for metric in METRICS}}


def main() -> None:
    rows = [
        load_row(ROOT / "outputs/baselines_seed2026/summary.csv", "text", 2026),
        load_row(ROOT / "outputs/exp001_text_bigru/summary.csv", "text_bigru", 2026),
    ]
    for seed in (2027, 2028):
        path = ROOT / f"outputs/exp005_seed{seed}/summary.csv"
        rows.extend(
            [
                load_row(path, "text", seed),
                load_row(path, "text_bigru", seed),
            ]
        )

    detail = pd.DataFrame(rows).sort_values(["seed", "model"])
    output_dir = ROOT / "outputs/exp005_multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "seed_results.csv", index=False, encoding="utf-8-sig")

    aggregate = detail.groupby("model")[METRICS].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(
        output_dir / "aggregate.csv", index=False, encoding="utf-8-sig"
    )

    wide = detail.pivot(index="seed", columns="model", values=METRICS)
    deltas = pd.DataFrame({"seed": wide.index})
    for metric in METRICS:
        deltas[f"{metric}_delta_bigru_minus_mean"] = (
            wide[(metric, "text_bigru")] - wide[(metric, "text")]
        ).to_numpy()
    deltas.to_csv(output_dir / "paired_deltas.csv", index=False, encoding="utf-8-sig")

    print("逐种子结果：")
    print(detail.to_string(index=False))
    print("\n均值与样本标准差：")
    print(aggregate.to_string())
    print("\n配对差值（MAE 为负表示 BiGRU 更好）：")
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
