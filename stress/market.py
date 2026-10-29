"""
Market risk, counterparty credit risk and CVA (adverse scenario; instantaneous shock booked in year 1).

* Full revaluation approximated by sensitivities: ΔFV = Σ sensitivity × shock (HFT through P&L, FVOCI through OCI —
  both reduce CET1).
* Prudent valuation: the AVA stock rises by a multiplier (higher uncertainty and close-out costs).
* Counterparty default: among the ten largest counterparties (excluding sovereigns and CCPs), the two most vulnerable
  (lowest ratings) default; loss = (stressed EAD − collateral)⁺ × LGD.
* CVA: increase of the CVA reserve for the widening of counterparty spreads (financials shock).
* Years 2-3: client trading revenues at a share of the starting level (haircut in the adverse scenario).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import RATING_ORDER


@dataclass
class MarketParams:
    ava_multiplier: float = 0.5        # AVA stock increase (adverse)
    n_defaults: int = 2
    trading_haircut: float = 0.5       # adverse client revenues in Y2-Y3 as (1 − haircut) × starting level
    cva_spread_factor: str = "Financials"


def revaluation(market: pd.DataFrame, shocks: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    m = market.merge(shocks[["risk type", "factor", "shock", "unit"]], on=["risk type", "factor"], how="left").fillna({"shock": 0.0})
    m["shock"] *= scale
    m["P&L"] = m["sensitivity"] * m["shock"]
    return m


def ccr_defaults(cpty: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    top = cpty.nlargest(10, "stressed EAD")
    elig = top[~top["type"].isin(["Sovereign", "CCP"])].copy()
    elig["rank"] = elig["rating"].map({r: i for i, r in enumerate(RATING_ORDER)})
    d = elig.sort_values(["rank", "stressed EAD"], ascending=[False, False]).head(n).copy()
    d["loss"] = np.maximum(d["stressed EAD"] - d["collateral"], 0) * d["LGD"]
    return d


def market_losses(bank, shocks: pd.DataFrame, P: MarketParams, scale: float = 1.0, adverse: bool = True) -> dict:
    if not adverse or scale == 0:
        return {"revaluation": revaluation(bank.market, shocks, 0.0), "HFT": 0.0, "FVOCI": 0.0, "AVA": 0.0, "CCR default": 0.0, "CVA": 0.0,
                "defaulted": ccr_defaults(bank.cpty, P.n_defaults).iloc[0:0], "trading income": [bank.trading_income] * 3}
    rv = revaluation(bank.market, shocks, scale)
    d = ccr_defaults(bank.cpty, P.n_defaults)
    sp = shocks.loc[(shocks["risk type"] == "Credit spread") & (shocks["factor"] == P.cva_spread_factor), "shock"]
    cva = bank.cva_cs01 * (float(sp.iloc[0]) if len(sp) else 0.0) * scale
    ti = [bank.trading_income * (1 - P.trading_haircut * min(scale, 1.0))] * 3
    ti[0] = 0.0                                                            # year 1: shock only, no client revenue assumed
    return {"revaluation": rv, "HFT": float(rv.loc[rv["portfolio"] == "HFT", "P&L"].sum()),
            "FVOCI": float(rv.loc[rv["portfolio"] == "FVOCI", "P&L"].sum()), "AVA": -bank.ava * P.ava_multiplier * scale,
            "CCR default": -float(d["loss"].sum()) * min(scale, 1.0) if scale > 0 else 0.0, "CVA": -cva, "defaulted": d, "trading income": ti}
