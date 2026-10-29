"""
Macro-financial scenarios (annual, three-year horizon, static balance sheet).

The default adverse scenario is *illustrative*: it is calibrated on the headline figures of the ESRB adverse scenario
of the 2025 EU-wide stress test (cumulative EU GDP fall of about 6.3 % over 2025-2027, unemployment about 6 pp above
baseline at the end of the horizon, inflation 5.0 % / 3.5 % / 1.9 %). House prices, commercial real estate, rates and
the instantaneous market shocks are plausible values chosen for the demo, not the official ESRB numbers.

Country severity: scenario_country = baseline + m_country × (adverse − baseline).
Scaling (reverse stress test): adverse(s) = baseline + s × (adverse − baseline), and market shocks × s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VARS = ["GDP growth", "unemployment", "HPI growth", "CRE growth", "inflation", "short rate", "long rate"]
START = 2024
YEARS = [2025, 2026, 2027]
COUNTRIES = ["FR", "DE", "IT", "ES", "Other EU"]
COUNTRY_MULT = pd.Series({"FR": 1.00, "DE": 1.00, "IT": 1.15, "ES": 1.10, "Other EU": 0.90})


def default_paths() -> dict[str, pd.DataFrame]:
    """Year 2024 = starting point (actual), then 2025-2027. Rates and growth in %, unemployment rate in %."""
    base = pd.DataFrame({
        "GDP growth": [0.9, 1.0, 1.2, 1.3], "unemployment": [6.0, 5.9, 5.8, 5.8], "HPI growth": [1.5, 2.0, 2.5, 3.0],
        "CRE growth": [-2.0, 1.0, 1.5, 2.0], "inflation": [2.4, 2.1, 1.9, 2.0], "short rate": [3.0, 2.2, 2.1, 2.2],
        "long rate": [2.7, 2.9, 3.0, 3.1]}, index=[START] + YEARS)
    adv = pd.DataFrame({
        "GDP growth": [0.9, -2.9, -2.5, -0.9], "unemployment": [6.0, 8.6, 10.7, 11.8], "HPI growth": [1.5, -7.5, -7.0, -3.5],
        "CRE growth": [-2.0, -11.0, -9.0, -4.0], "inflation": [2.4, 5.0, 3.5, 1.9], "short rate": [3.0, 3.6, 3.2, 2.6],
        "long rate": [2.7, 4.0, 3.8, 3.5]}, index=[START] + YEARS)
    return {"baseline": base, "adverse": adv}


def default_market_shocks() -> pd.DataFrame:
    """Instantaneous shocks applied to fair-value positions at the start of year 1 (adverse); baseline = 0."""
    return pd.DataFrame([
        ("Interest rates", "EUR 2y", 90.0, "bp"), ("Interest rates", "EUR 10y", 110.0, "bp"), ("Interest rates", "USD 10y", 80.0, "bp"),
        ("Sovereign spread", "FR", 80.0, "bp"), ("Sovereign spread", "DE", 25.0, "bp"), ("Sovereign spread", "IT", 210.0, "bp"),
        ("Sovereign spread", "ES", 160.0, "bp"), ("Credit spread", "IG corporate", 140.0, "bp"), ("Credit spread", "HY corporate", 450.0, "bp"),
        ("Credit spread", "Financials", 220.0, "bp"), ("Equity", "EU equity", -38.0, "%"), ("Equity", "US equity", -32.0, "%"),
        ("FX", "USD/EUR", -12.0, "%"), ("Commodity", "Energy", 35.0, "%"),
    ], columns=["risk type", "factor", "shock", "unit"])


def country_path(base: pd.DataFrame, adv: pd.DataFrame, country: str, scale: float = 1.0, mult: pd.Series | None = None) -> pd.DataFrame:
    m = (COUNTRY_MULT if mult is None else mult).get(country, 1.0) * scale
    out = base + m * (adv - base)
    out.loc[START] = base.loc[START]
    return out


def scaled(paths: dict, scale: float) -> pd.DataFrame:
    out = paths["baseline"] + scale * (paths["adverse"] - paths["baseline"])
    out.loc[START] = paths["baseline"].loc[START]
    return out


def cumulative(path: pd.DataFrame, var: str) -> pd.Series:
    """Index level (start = 100) of a growth variable."""
    g = path[var].iloc[1:] / 100
    return 100 * (1 + g).cumprod()


def regressors(path: pd.DataFrame) -> pd.DataFrame:
    """Variables used by the satellite models for each projection year."""
    out = pd.DataFrame(index=path.index[1:])
    out["GDP growth"] = path["GDP growth"].iloc[1:].to_numpy()
    out["Δ unemployment"] = np.diff(path["unemployment"].to_numpy())
    out["HPI growth"] = path["HPI growth"].iloc[1:].to_numpy()
    out["CRE growth"] = path["CRE growth"].iloc[1:].to_numpy()
    out["short rate"] = path["short rate"].iloc[1:].to_numpy()
    return out


def start_regressors(path: pd.DataFrame, prev_unemployment: float = 6.3) -> pd.Series:
    p = path.loc[START]
    return pd.Series({"GDP growth": p["GDP growth"], "Δ unemployment": p["unemployment"] - prev_unemployment,
                      "HPI growth": p["HPI growth"], "CRE growth": p["CRE growth"], "short rate": p["short rate"]})
