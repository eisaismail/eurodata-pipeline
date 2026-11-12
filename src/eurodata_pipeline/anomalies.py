"""Compare unusual daily moves using chronological ECB observations."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["daily_change_pct", "signed_z", "volatility_ratio"]


def features(frame):
    required = {"currency", "observation_date", "rate", "base_currency"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing exchange-rate columns")
    frame = frame.copy()
    frame["observation_date"] = pd.to_datetime(frame.observation_date, errors="raise")
    if (frame.rate <= 0).any() or not np.isfinite(frame.rate).all():
        raise ValueError("Rates must be positive and finite")
    if frame.duplicated(["currency", "observation_date"]).any():
        raise ValueError("Duplicate currency dates")
    if not frame.base_currency.eq("EUR").all():
        raise ValueError("Expected EUR reference rates")
    groups = []
    for _, group in frame.groupby("currency"):
        group = group.sort_values("observation_date").copy()
        returns = group.rate.pct_change(fill_method=None) * 100
        past = returns.shift(1)
        std = past.rolling(60, min_periods=60).std()
        group["daily_change_pct"] = returns
        group["signed_z"] = (returns - past.rolling(60).mean()) / std
        group["volatility_ratio"] = past.rolling(20).std() / std
        groups.append(group)
    return pd.concat(groups).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def score(frame):
    scored, summaries, details = [], [], {}
    for currency, group in features(frame).groupby("currency"):
        train = group[group.observation_date < "2023-01-01"]
        calibration = group[group.observation_date.between("2023-01-01", "2023-12-31")]
        test = group[group.observation_date >= "2024-01-01"].copy()
        if len(train) < 250 or len(calibration) < 100 or len(test) < 30:
            raise ValueError(f"Insufficient chronological history for {currency}")
        model = IsolationForest(n_estimators=300, max_samples=256, random_state=42,
                                contamination="auto", n_jobs=1)
        model.fit(train[FEATURES])
        baseline_threshold = float(np.quantile(calibration.signed_z.abs(), .98, method="higher"))
        ml_threshold = float(np.quantile(-model.score_samples(calibration[FEATURES]),
                                         .98, method="higher"))
        test["baseline_score"] = test.signed_z.abs()
        test["ml_score"] = -model.score_samples(test[FEATURES])
        test["baseline_flag"] = test.baseline_score > baseline_threshold
        test["ml_flag"] = test.ml_score > ml_threshold
        both = test.baseline_flag & test.ml_flag
        summaries.append({"currency": currency, "test_observations": len(test),
                          "baseline_flags": int(test.baseline_flag.sum()),
                          "ml_flags": int(test.ml_flag.sum()), "both_flags": int(both.sum()),
                          "baseline_threshold": baseline_threshold, "ml_threshold": ml_threshold})
        details[currency] = {"training_rows": len(train), "calibration_rows": len(calibration),
                             "test_rows": len(test), "training_last_date": str(train.observation_date.max().date()),
                             "test_last_date": str(test.observation_date.max().date())}
        scored.append(test)
    return pd.concat(scored, ignore_index=True), pd.DataFrame(summaries), details


def write_report(frame, destination, source_path):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scored, summary, details = score(frame)
    scored.to_csv(destination / "scored_observations.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    metadata = {"source": "European Central Bank EXR daily reference rates",
                "source_csv_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                "training": "Through 2022-12-31", "calibration": "2023 calendar year",
                "test": "2024-01-01 onwards", "threshold": "2023 98th percentile, strict exceedance",
                "features": FEATURES, "random_state": 42, "n_estimators": 300,
                "max_samples": 256, "labelled_anomalies": False, "currencies": details}
    metadata["package_versions"] = {name: version(name) for name in
                                    ("pandas", "numpy", "scikit-learn", "matplotlib")}
    (destination / "method.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (currency, group) in zip(axes, scored.groupby("currency")):
        ax.plot(group.observation_date, group.daily_change_pct, color="#aab4c0", lw=.7)
        for mask, label, color in [
            (group.baseline_flag & ~group.ml_flag, "Statistical rule only", "#dc8b13"),
            (group.ml_flag & ~group.baseline_flag, "ML only", "#7653bc"),
            (group.ml_flag & group.baseline_flag, "Both", "#c63242")]:
            ax.scatter(group.loc[mask, "observation_date"], group.loc[mask, "daily_change_pct"],
                       s=18, label=label, color=color, zorder=3)
        ax.set_ylabel(f"{currency}\nchange (%)")
        ax.grid(alpha=.2)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Unusual daily ECB reference-rate changes: held-out dates from 2024")
    fig.text(.5, .01, "Flags indicate unusual observations, not confirmed errors or trading advice.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(destination / "daily_changes.svg")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/ecb_reference_rates.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/anomalies"))
    args = parser.parse_args()
    print(write_report(pd.read_csv(args.input), args.output, args.input).to_string(index=False))


if __name__ == "__main__":
    main()
