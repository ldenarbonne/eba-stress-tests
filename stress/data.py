"""
Synthetic mid-sized EU bank at the starting point (31 December 2024), and CSV validation.

* credit: one row per portfolio × country with stage 1/2/3 exposures (EAD), PD (point in time, 12 months), regulatory
  PD (through the cycle), LGD, stage-3 coverage, approach (IRB / SA), SA risk weight, maturity, client margin and the
  fixed-rate share (for NII).
* market: fair-value sensitivities (held-for-trading and FVOCI) by risk factor, prudent-valuation reserves (AVA),
  CVA and its spread sensitivity; counterparties (top 10) for the counterparty-default shock.
* funding: interest-bearing liabilities and non-loan assets for NII (pass-through β to the short rate, spread, fixed share).
* pnl: starting-point P&L items and historical operational / conduct losses.
* capital: CET1, AT1, T2, REA for market, CCR + CVA and operational risk, leverage exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .scenario import COUNTRIES

PORTFOLIOS = pd.DataFrame({
    "approach": ["SA", "IRB", "IRB", "IRB", "SA", "IRB", "SA"],
    "irb class": ["-", "institution", "corporate", "sme", "-", "mortgage", "-"],
    "size €bn": [26.0, 12.0, 52.0, 24.0, 14.0, 62.0, 16.0],
    "pd": [0.0010, 0.0015, 0.0080, 0.0140, 0.0150, 0.0060, 0.0220],
    "pd_reg": [0.0012, 0.0020, 0.0060, 0.0110, 0.0200, 0.0045, 0.0300],
    "lgd": [0.45, 0.45, 0.38, 0.35, 0.26, 0.17, 0.62],
    "s2 share": [0.01, 0.02, 0.09, 0.12, 0.13, 0.07, 0.10],
    "s3 share": [0.001, 0.002, 0.030, 0.045, 0.050, 0.017, 0.040],
    "s3 coverage": [0.40, 0.45, 0.48, 0.45, 0.35, 0.24, 0.66],
    "rw_sa": [0.05, 0.40, 0.95, 0.80, 1.00, 0.30, 0.75],
    "maturity": [5.0, 2.0, 3.0, 3.0, 5.0, 15.0, 3.0],
    "margin": [0.0030, 0.0040, 0.0140, 0.0200, 0.0190, 0.0110, 0.0420],
    "fixed share": [0.90, 0.30, 0.35, 0.40, 0.30, 0.75, 0.85],
}, index=["Sovereign", "Institutions", "Corporates", "SME", "Commercial real estate", "Residential mortgages", "Consumer credit"])

COUNTRY_SPLIT = pd.Series({"FR": 0.45, "DE": 0.15, "IT": 0.15, "ES": 0.12, "Other EU": 0.13})
COUNTRY_RISK = pd.Series({"FR": 1.00, "DE": 0.85, "IT": 1.30, "ES": 1.15, "Other EU": 0.95})

CREDIT_COLS = ["portfolio", "country", "approach", "irb_class", "ead_s1", "ead_s2", "ead_s3", "pd", "pd_reg", "lgd", "s3_coverage",
               "rw_sa", "maturity", "margin", "fixed_share"]


def credit_book(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p, r in PORTFOLIOS.iterrows():
        for c in COUNTRIES:
            ead = r["size €bn"] * 1e9 * COUNTRY_SPLIT[c] * rng.uniform(0.85, 1.15)
            k = COUNTRY_RISK[c] * rng.uniform(0.9, 1.1)
            s3 = min(r["s3 share"] * k, 0.2)
            s2 = min(r["s2 share"] * k, 0.4)
            rows.append({"portfolio": p, "country": c, "approach": r["approach"], "irb_class": r["irb class"],
                         "ead_s1": ead * (1 - s2 - s3), "ead_s2": ead * s2, "ead_s3": ead * s3,
                         "pd": r["pd"] * k, "pd_reg": r["pd_reg"] * k, "lgd": r["lgd"], "s3_coverage": r["s3 coverage"],
                         "rw_sa": r["rw_sa"] if p != "Sovereign" else {"FR": 0.0, "DE": 0.0, "IT": 0.20, "ES": 0.0, "Other EU": 0.10}[c],
                         "maturity": r["maturity"], "margin": r["margin"] * (1 + 0.15 * (k - 1)), "fixed_share": r["fixed share"]})
    return pd.DataFrame(rows)


def market_book() -> pd.DataFrame:
    """Sensitivity in € per unit of shock (per bp, or per 1 %). Portfolio: HFT or FVOCI."""
    return pd.DataFrame([
        ("Interest rates", "EUR 2y", "HFT", -0.35e6), ("Interest rates", "EUR 10y", "HFT", -0.55e6), ("Interest rates", "USD 10y", "HFT", 0.20e6),
        ("Interest rates", "EUR 10y", "FVOCI", -1.10e6),
        ("Sovereign spread", "FR", "FVOCI", -1.20e6), ("Sovereign spread", "DE", "FVOCI", -0.40e6), ("Sovereign spread", "IT", "FVOCI", -0.80e6),
        ("Sovereign spread", "ES", "FVOCI", -0.70e6), ("Sovereign spread", "IT", "HFT", -0.15e6),
        ("Credit spread", "IG corporate", "HFT", -0.55e6), ("Credit spread", "HY corporate", "HFT", -0.08e6), ("Credit spread", "Financials", "FVOCI", -0.45e6),
        ("Equity", "EU equity", "HFT", 5.0e6), ("Equity", "US equity", "HFT", 2.0e6), ("Equity", "EU equity", "FVOCI", 3.0e6),
        ("FX", "USD/EUR", "HFT", 3.0e6), ("Commodity", "Energy", "HFT", -1.5e6),
    ], columns=["risk type", "factor", "portfolio", "sensitivity"])


def counterparties() -> pd.DataFrame:
    return pd.DataFrame([
        ("Bank A", "Bank", "A", 1.35e9, 1.10e9, 0.60), ("Bank B", "Bank", "A-", 1.10e9, 0.80e9, 0.60), ("Bank C", "Bank", "BBB+", 0.80e9, 0.45e9, 0.60),
        ("Hedge fund D", "Fund", "BB", 0.75e9, 0.40e9, 0.60), ("Insurer E", "Insurer", "A", 0.70e9, 0.30e9, 0.60),
        ("Corporate F", "Corporate", "BBB-", 0.65e9, 0.05e9, 0.45), ("Corporate G", "Corporate", "BB+", 0.55e9, 0.00, 0.45),
        ("Sovereign H", "Sovereign", "AA", 1.60e9, 0.00, 0.45), ("CCP I", "CCP", "AA", 2.20e9, 2.00e9, 0.60),
        ("Pension fund J", "Fund", "AA-", 0.50e9, 0.35e9, 0.60),
    ], columns=["counterparty", "type", "rating", "stressed EAD", "collateral", "LGD"])


RATING_ORDER = ["AA", "AA-", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B"]

FUNDING = pd.DataFrame([
    # name, side, amount, beta (pass-through of short-rate changes), spread over short rate (or fixed rate), fixed share, repricing per year
    ("Central bank reserves", "asset", 14e9, 1.00, -0.0000, 0.00, 1.0),
    ("Debt securities (HQLA)", "asset", 30e9, 1.00, 0.0020, 0.80, 0.20),
    ("Retail sight deposits", "liability", 78e9, 0.15, -0.0000, 0.00, 1.0),
    ("Retail term deposits", "liability", 40e9, 0.75, 0.0010, 0.30, 0.50),
    ("Corporate deposits", "liability", 46e9, 0.55, 0.0000, 0.00, 1.0),
    ("Covered bonds", "liability", 22e9, 1.00, 0.0040, 0.90, 0.15),
    ("Senior unsecured debt", "liability", 26e9, 1.00, 0.0110, 0.40, 0.25),
    ("Central bank & interbank funding", "liability", 12e9, 1.00, 0.0005, 0.00, 1.0),
], columns=["item", "side", "amount", "beta", "spread", "fixed share", "repricing"])


@dataclass
class Bank:
    credit: pd.DataFrame
    market: pd.DataFrame = field(default_factory=market_book)
    cpty: pd.DataFrame = field(default_factory=counterparties)
    funding: pd.DataFrame = field(default_factory=lambda: FUNDING.copy())
    # P&L at the starting point (2024, € per year)
    fees: float = 2.05e9
    expenses: float = 3.95e9
    trading_income: float = 0.45e9
    other_income: float = 0.10e9
    oprisk_hist: float = 0.24e9          # average non-conduct operational losses (last 5 years)
    conduct_hist: float = 0.09e9         # average conduct losses (last 5 years)
    material_conduct: float = 0.15e9     # identified material conduct events, booked over the horizon
    ava: float = 0.32e9
    cva: float = 0.18e9
    cva_cs01: float = 1.2e6              # CVA increase per bp of counterparty spread widening
    # capital
    cet1: float = 19.0e9
    at1: float = 1.5e9
    t2: float = 2.5e9
    rea_market: float = 4.2e9
    rea_ccr_cva: float = 3.6e9
    rea_op: float = 10.5e9
    lev_exposure: float = 318e9
    tax: float = 0.30                    # EBA common tax rate assumption (illustrative)
    payout: float = 0.50


def generate(seed: int = 3) -> Bank:
    return Bank(credit=credit_book(seed))


def validate_credit(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in CREDIT_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"credit file: missing columns {', '.join(miss)}")
    df = df[CREDIT_COLS].copy()
    for c in CREDIT_COLS[4:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[CREDIT_COLS[4:]].isna().any().any():
        raise ValueError("credit file: non-numeric or missing values")
    if not set(df["portfolio"]) <= set(PORTFOLIOS.index):
        raise ValueError(f"portfolios must be in {list(PORTFOLIOS.index)}")
    if not set(df["approach"]) <= {"IRB", "SA"}:
        raise ValueError("approach must be IRB or SA")
    if ((df[["pd", "pd_reg", "lgd", "s3_coverage"]] < 0) | (df[["pd", "pd_reg", "lgd", "s3_coverage"]] > 1)).any().any():
        raise ValueError("pd, pd_reg, lgd and s3_coverage must be between 0 and 1")
    return df.reset_index(drop=True)
