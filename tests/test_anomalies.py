"""Checks against published ECB observations, with no injected rate spikes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurodata_pipeline.anomalies import features, score, write_report

SOURCE = Path(__file__).resolve().parents[1] / "data/ecb_reference_rates.csv"


@pytest.fixture(scope="module")
def rates():
    return pd.read_csv(SOURCE)


def test_features_use_only_available_history(rates):
    cutoff = "2024-06-28"
    full = features(rates)
    prefix = features(rates[rates.observation_date <= cutoff])
    pd.testing.assert_frame_equal(full[full.observation_date <= cutoff], prefix)
    usd = rates[rates.currency == "USD"].sort_values("observation_date")
    last = features(usd).iloc[-1]
    assert last.daily_change_pct == pytest.approx((usd.rate.iloc[-1] / usd.rate.iloc[-2] - 1) * 100)


def test_future_dates_do_not_change_models_or_thresholds(rates):
    full, summary, _ = score(rates)
    prefix, old_summary, details = score(rates[rates.observation_date <= "2024-06-28"])
    pd.testing.assert_frame_equal(full[full.observation_date <= "2024-06-28"].reset_index(drop=True),
                                  prefix.reset_index(drop=True))
    np.testing.assert_allclose(summary[["baseline_threshold", "ml_threshold"]],
                               old_summary[["baseline_threshold", "ml_threshold"]])
    assert all(d["training_last_date"] < "2023-01-01" for d in details.values())
    assert prefix.observation_date.min() >= pd.Timestamp("2024-01-01")


def test_report_is_reproducible_and_uses_real_rates(rates, tmp_path):
    summary = write_report(rates, tmp_path, SOURCE)
    saved = pd.read_csv(tmp_path / "scored_observations.csv")
    joined = saved.merge(rates, on=["currency", "observation_date"], suffixes=("", "_source"))
    assert len(joined) == len(saved)
    np.testing.assert_allclose(joined.rate, joined.rate_source)
    assert set(summary.currency) == {"CHF", "GBP", "JPY", "USD"}
    assert (tmp_path / "daily_changes.svg").exists()
    assert (tmp_path / "method.json").exists()


def test_insufficient_history_is_rejected(rates):
    with pytest.raises(ValueError, match="Insufficient"):
        score(rates[rates.observation_date >= "2024-01-01"])
