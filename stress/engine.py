"""
Projection engine: P&L, capital, REA and ratios for the baseline and the adverse scenario (three years, static
balance sheet), plus the decomposition of the CET1 ratio change.

P&L (each year)
  NII + net fees & commissions + trading income + other income − administrative expenses − impairments
  − operational and conduct losses (+ market shock in year 1 for HFT positions) = profit before tax;
  tax at a common rate on positive profits (no tax credit on losses unless the option is set);
  distributions = payout × positive net profit.
CET1_t = CET1_{t−1} + net profit − distributions + OCI (FVOCI revaluation, after tax) − ΔAVA − ΔIRB shortfall.
REA = credit REA (floored at its starting value) + market + CCR/CVA + operational (held constant);
  output floor (CRR3 art. 92(3) and 465): transitional 50 % / 55 % / 60 % in 2025 / 2026 / 2027, fully loaded 72.5 %.
Leverage ratio = Tier 1 / leverage exposure (static).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import credit as CR
from . import market as MK
from . import nii as NI
from . import scenario as SC

FLOOR_TRANSITIONAL = {2024: 0.0, 2025: 0.50, 2026: 0.55, 2027: 0.60, 2028: 0.65, 2029: 0.70, 2030: 0.725}
BLOCKS = ["credit", "market", "nii", "oprisk", "fees & expenses"]


@dataclass
class Params:
    credit: CR.CreditParams = field(default_factory=CR.CreditParams)
    market: MK.MarketParams = field(default_factory=MK.MarketParams)
    nii: NI.NIIParams = field(default_factory=NI.NIIParams)
    fee_haircut: float = 0.12          # adverse fee & commission income reduction vs starting point
    op_multiplier: float = 1.6         # adverse non-conduct operational losses vs historical average
    conduct_multiplier: float = 1.3    # adverse non-material conduct losses vs historical average
    expenses_inflation: bool = True    # expenses grow with scenario inflation, floored at the starting level
    dta_on_losses: bool = False
    output_floor: bool = True
    country_mult: pd.Series | None = None


@dataclass
class Result:
    scenario: str
    pnl: pd.DataFrame
    capital: pd.DataFrame
    credit: pd.DataFrame
    nii: pd.DataFrame
    market: dict
    waterfall: pd.Series | None = None


def _blocks(scenario, blocks):
    if scenario == "baseline":
        return {b: False for b in BLOCKS}
    if blocks is None:
        return {b: True for b in BLOCKS}
    return {b: (b in blocks) for b in BLOCKS}


def run(bank, paths: dict, shocks: pd.DataFrame, fits: dict, P: Params, scenario: str = "adverse", scale: float = 1.0,
        blocks: list | None = None) -> Result:
    on = _blocks(scenario, blocks)
    eu_adv = SC.scaled(paths, scale)
    eu = eu_adv if scenario == "adverse" else paths["baseline"]
    # credit
    cp = CR.project(bank.credit, paths, fits, P.credit, "adverse" if on["credit"] else "baseline", scale, P.country_mult)
    # NII (rates path of the euro area aggregate scenario)
    nii = NI.project(bank, cp, eu_adv if on["nii"] else paths["baseline"], P.nii, on["nii"], scale)
    mk = MK.market_losses(bank, shocks, P.market, scale, on["market"])
    years = SC.YEARS
    rows = []
    cet1 = bank.cet1
    infl = (eu_adv if on["fees & expenses"] else paths["baseline"])["inflation"]
    cum_infl = 1.0
    cg = cp.groupby("year")
    shortfall0 = max(cg.get_group(SC.START)["EL IRB"].sum() - cg.get_group(SC.START)["prov IRB perf"].sum(), 0)
    sf_prev = shortfall0
    cred_rea0 = cg.get_group(SC.START)["REA"].sum()
    sa_eq0 = cg.get_group(SC.START)["REA SA-equivalent"].sum()
    other_rea = bank.rea_market + bank.rea_ccr_cva + bank.rea_op
    cap_rows = [{"year": SC.START, "CET1": cet1, "credit REA": cred_rea0, "REA SA-equivalent": sa_eq0 + other_rea}]
    for i, y in enumerate(years):
        g = cg.get_group(y)
        cum_infl *= 1 + infl.loc[y] / 100
        fees = bank.fees * (1 - P.fee_haircut * min(scale, 1.5)) if on["fees & expenses"] else bank.fees
        exp_ = bank.expenses * max(cum_infl, 1.0) if P.expenses_inflation else bank.expenses
        op = bank.oprisk_hist * ((1 + (P.op_multiplier - 1) * scale) if on["oprisk"] else 1.0)
        conduct = bank.conduct_hist * ((1 + (P.conduct_multiplier - 1) * scale) if on["oprisk"] else 1.0) \
            + (bank.material_conduct if (on["oprisk"] and i == 0) else bank.material_conduct / 3 if not on["oprisk"] else 0.0)
        trading = mk["trading income"][i] if on["market"] else bank.trading_income
        shock_pl = (mk["HFT"] + mk["CCR default"] + mk["CVA"]) if (on["market"] and i == 0) else 0.0
        imp = g["impairment"].sum()
        nii_y = nii.loc[y, "NII"]
        pbt = nii_y + fees + trading + bank.other_income + shock_pl - exp_ - imp - op - conduct
        tax = bank.tax * pbt if (pbt > 0 or P.dta_on_losses) else 0.0
        np_ = pbt - tax
        div = bank.payout * max(np_, 0.0)
        oci = mk["FVOCI"] * ((1 - bank.tax) if P.dta_on_losses else 1.0) if (on["market"] and i == 0) else 0.0
        d_ava = mk["AVA"] if (on["market"] and i == 0) else 0.0
        sf = max(g["EL IRB"].sum() - g["prov IRB perf"].sum(), 0)
        d_sf = -(sf - sf_prev)
        sf_prev = sf
        cet1 = cet1 + np_ - div + oci + d_ava + d_sf
        cred_rea = g["REA"].sum()
        if P.credit.rea_floor_start:
            cred_rea = max(cred_rea, cred_rea0)
        rows.append({"year": y, "net interest income": nii_y, "net fees & commissions": fees, "trading income": trading,
                     "other income": bank.other_income, "market shock (HFT, CCR, CVA)": shock_pl, "administrative expenses": -exp_,
                     "impairments": -imp, "operational losses": -op, "conduct losses": -conduct, "profit before tax": pbt, "tax": -tax,
                     "net profit": np_, "distributions": -div})
        cap_rows.append({"year": y, "CET1": cet1, "OCI (FVOCI)": oci, "AVA": d_ava, "IRB shortfall": d_sf,
                         "credit REA": cred_rea, "REA SA-equivalent": g["REA SA-equivalent"].sum() + other_rea})
    pnl = pd.DataFrame(rows).set_index("year")
    cap = pd.DataFrame(cap_rows).set_index("year").fillna(0.0)
    cap["market REA"] = bank.rea_market
    cap["CCR & CVA REA"] = bank.rea_ccr_cva
    cap["operational REA"] = bank.rea_op
    cap["REA (pre-floor)"] = cap[["credit REA", "market REA", "CCR & CVA REA", "operational REA"]].sum(axis=1)
    if P.output_floor:
        tf = cap.index.map(lambda y: FLOOR_TRANSITIONAL.get(y, 0.725))
        cap["REA transitional"] = np.maximum(cap["REA (pre-floor)"], tf * cap["REA SA-equivalent"])
        cap["REA fully loaded"] = np.maximum(cap["REA (pre-floor)"], 0.725 * cap["REA SA-equivalent"])
    else:
        cap["REA transitional"] = cap["REA fully loaded"] = cap["REA (pre-floor)"]
    cap["Tier 1"] = cap["CET1"] + bank.at1
    cap["CET1 ratio (transitional)"] = cap["CET1"] / cap["REA transitional"]
    cap["CET1 ratio (fully loaded)"] = cap["CET1"] / cap["REA fully loaded"]
    cap["Tier 1 ratio"] = cap["Tier 1"] / cap["REA transitional"]
    cap["total capital ratio"] = (cap["Tier 1"] + bank.t2) / cap["REA transitional"]
    cap["leverage ratio"] = cap["Tier 1"] / bank.lev_exposure
    res = Result(scenario, pnl, cap, cp, nii, mk)
    res.waterfall = waterfall(res)
    return res


def waterfall(res: Result, ratio: str = "CET1 ratio (fully loaded)", rea_col: str = "REA fully loaded") -> pd.Series:
    """Change of the CET1 ratio from the start to the end of the horizon, by driver (in ratio points)."""
    cap, pnl = res.capital, res.pnl
    rea0 = cap[rea_col].iloc[0]
    pbt = pnl["profit before tax"]
    tax = pnl["tax"]
    items = {
        "net interest income": pnl["net interest income"].sum(),
        "fees & commissions": pnl["net fees & commissions"].sum(),
        "trading & other income": pnl["trading income"].sum() + pnl["other income"].sum(),
        "market risk & CCR": pnl["market shock (HFT, CCR, CVA)"].sum() + cap["OCI (FVOCI)"].sum() + cap["AVA"].sum(),
        "impairments (credit risk)": pnl["impairments"].sum(),
        "operational & conduct": pnl["operational losses"].sum() + pnl["conduct losses"].sum(),
        "administrative expenses": pnl["administrative expenses"].sum(),
        "tax": tax.sum(),
        "distributions": pnl["distributions"].sum(),
        "IRB shortfall & other": cap["IRB shortfall"].sum(),
    }
    s = pd.Series(items) / rea0
    end_ratio = cap[ratio].iloc[-1]
    s["REA increase"] = end_ratio - cap["CET1"].iloc[-1] / rea0
    s.attrs.update(start=cap[ratio].iloc[0], end=end_ratio, low=cap[ratio].min())
    return s


def contributions(bank, paths, shocks, fits, P: Params, ratio: str = "CET1 ratio (fully loaded)") -> pd.Series:
    """One-at-a-time: depletion when only one block is stressed, minus the baseline change."""
    base = run(bank, paths, shocks, fits, P, "baseline").capital[ratio]
    out = {}
    for b in BLOCKS:
        c = run(bank, paths, shocks, fits, P, "adverse", 1.0, [b]).capital[ratio]
        out[b] = (c.iloc[-1] - base.iloc[-1])
    full = run(bank, paths, shocks, fits, P, "adverse").capital[ratio]
    out["interaction"] = (full.iloc[-1] - base.iloc[-1]) - sum(out.values())
    return pd.Series(out)


def reverse_stress(bank, paths, shocks, fits, P: Params, threshold: float, ratio: str = "CET1 ratio (fully loaded)",
                   lo: float = 0.0, hi: float = 4.0) -> dict:
    f = lambda s: run(bank, paths, shocks, fits, P, "adverse", s).capital[ratio].min() - threshold  # noqa: E731
    grid = np.linspace(lo, hi, 17)
    vals = [f(s) + threshold for s in grid]
    if f(lo) < 0:
        return {"scale": 0.0, "grid": grid, "values": vals, "note": "threshold already breached without stress"}
    if f(hi) > 0:
        return {"scale": np.inf, "grid": grid, "values": vals, "note": f"threshold not reached up to {hi:.0f}× the adverse deviation"}
    s = brentq(f, lo, hi, xtol=1e-3)
    return {"scale": s, "grid": grid, "values": vals, "note": ""}


def tornado(bank, paths, shocks, fits, P: Params, ratio: str = "CET1 ratio (fully loaded)") -> pd.DataFrame:
    ref = run(bank, paths, shocks, fits, P).capital[ratio].iloc[-1]
    specs = [
        ("SICR transfer multiple ±40 %", lambda p, k: replace(p, credit=replace(p.credit, m_sicr=p.credit.m_sicr * (1 + 0.4 * k)))),
        ("Stage-2 PD multiple ±30 %", lambda p, k: replace(p, credit=replace(p.credit, m_s2=p.credit.m_s2 * (1 + 0.3 * k)))),
        ("Unsecured LGD sensitivity ±50 %", lambda p, k: replace(p, credit=replace(p.credit, lgd_lambda=p.credit.lgd_lambda * (1 + 0.5 * k)))),
        ("Regulatory PD cyclicality ±0.3", lambda p, k: replace(p, credit=replace(p.credit, elasticity=float(np.clip(p.credit.elasticity + 0.3 * k, 0, 1))))),
        ("Funding spread shock ±50 %", lambda p, k: replace(p, nii=replace(p.nii, funding_spread_shock=p.nii.funding_spread_shock * (1 + 0.5 * k)))),
        ("Fee haircut ±50 %", lambda p, k: replace(p, fee_haircut=p.fee_haircut * (1 + 0.5 * k))),
        ("Operational loss multiplier ±0.4", lambda p, k: replace(p, op_multiplier=p.op_multiplier + 0.4 * k)),
        ("AVA multiplier ±50 %", lambda p, k: replace(p, market=replace(p.market, ava_multiplier=p.market.ava_multiplier * (1 + 0.5 * k)))),
    ]
    rows = []
    for name, fn in specs:
        lo_ = run(bank, paths, shocks, fits, fn(P, -1)).capital[ratio].iloc[-1] - ref
        hi_ = run(bank, paths, shocks, fits, fn(P, 1)).capital[ratio].iloc[-1] - ref
        rows.append({"parameter": name, "low": lo_, "high": hi_})
    for name, sc in [("Market shocks ±25 %", 0.25)]:
        rows.append({"parameter": name,
                     "low": run(bank, paths, shocks.assign(shock=shocks["shock"] * (1 - sc)), fits, P).capital[ratio].iloc[-1] - ref,
                     "high": run(bank, paths, shocks.assign(shock=shocks["shock"] * (1 + sc)), fits, P).capital[ratio].iloc[-1] - ref})
    df = pd.DataFrame(rows).set_index("parameter")
    df["range"] = (df["high"] - df["low"]).abs()
    return df.sort_values("range")
