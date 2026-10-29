"""
Result tables in the spirit of the EBA stress-test templates (simplified, not the official layouts):
CSV_SUM (summary), CSV_CAP (capital), CSV_P&L, CSV_CR_SCEN (risk parameters), CSV_CR_PROV (stages, flows, provisions),
CSV_CR_REA, CSV_MR (market risk, CCR, CVA, AVA), CSV_NII, CSV_OR. Amounts in € million.
"""
from __future__ import annotations

import io

import pandas as pd

from . import scenario as SC

MN = 1e6


def _long(res_b, res_a, fn):
    out = pd.concat([fn(res_b).assign(scenario="baseline"), fn(res_a).assign(scenario="adverse")])
    return out[["scenario"] + [c for c in out.columns if c != "scenario"]]


def summary(res_b, res_a) -> pd.DataFrame:
    rows = []
    for r in (res_b, res_a):
        c = r.capital
        for col in ["CET1 ratio (transitional)", "CET1 ratio (fully loaded)", "Tier 1 ratio", "total capital ratio", "leverage ratio"]:
            rows.append({"scenario": r.scenario, "measure": col, "start": c[col].iloc[0], "end": c[col].iloc[-1], "lowest": c[col].min(),
                         "change start→end (bp)": (c[col].iloc[-1] - c[col].iloc[0]) * 1e4,
                         "max depletion (bp)": (c[col].iloc[0] - c[col].min()) * 1e4})
    return pd.DataFrame(rows)


def capital(res_b, res_a) -> pd.DataFrame:
    cols = ["CET1", "Tier 1", "credit REA", "market REA", "CCR & CVA REA", "operational REA", "REA (pre-floor)", "REA SA-equivalent",
            "REA transitional", "REA fully loaded"]
    f = lambda r: (r.capital[cols] / MN).join(r.capital[["CET1 ratio (transitional)", "CET1 ratio (fully loaded)", "leverage ratio"]]).reset_index()  # noqa: E731
    return _long(res_b, res_a, f)


def pnl(res_b, res_a) -> pd.DataFrame:
    return _long(res_b, res_a, lambda r: (r.pnl / MN).reset_index())


def credit_params(res_b, res_a) -> pd.DataFrame:
    def f(r):
        c = r.credit
        g = c.groupby(["portfolio", "year"])
        df = pd.DataFrame({"EAD": g["EAD"].sum() / MN, "new defaults": g["new defaults"].sum() / MN})
        perf_prev = (c.assign(perf=c["S1"] + c["S2"]).groupby(["portfolio", "year"])["perf"].sum()).groupby(level=0).shift(1)
        df["default rate"] = df["new defaults"] * MN / perf_prev
        w = c.assign(w=c["S1"] + c["S2"])
        df["PD (12m, avg)"] = (w["PD"] * w["w"]).groupby([w["portfolio"], w["year"]]).sum() / w.groupby(["portfolio", "year"])["w"].sum()
        df["LGD (avg)"] = (w["LGD"] * w["w"]).groupby([w["portfolio"], w["year"]]).sum() / w.groupby(["portfolio", "year"])["w"].sum()
        df["S3 coverage"] = g["prov S3"].sum() / g["S3"].sum()
        return df.reset_index()
    return _long(res_b, res_a, f)


def credit_provisions(res_b, res_a) -> pd.DataFrame:
    cols = ["S1", "S2", "S3", "S1→S2", "S2→S1", "new defaults", "prov S1", "prov S2", "prov S3", "provisions", "impairment"]
    return _long(res_b, res_a, lambda r: (r.credit.groupby(["portfolio", "year"])[cols].sum() / MN).reset_index())


def credit_rea(res_b, res_a) -> pd.DataFrame:
    return _long(res_b, res_a, lambda r: (r.credit.groupby(["portfolio", "approach", "year"])[["EAD", "REA", "REA SA-equivalent"]].sum() / MN).reset_index())


def market(res_a) -> pd.DataFrame:
    rv = res_a.market["revaluation"].copy()
    rv["P&L"] /= MN
    extra = pd.DataFrame([
        {"risk type": "Prudent valuation", "factor": "AVA increase", "portfolio": "CET1 deduction", "P&L": res_a.market["AVA"] / MN},
        {"risk type": "Counterparty credit risk", "factor": "default of the 2 most vulnerable counterparties", "portfolio": "P&L",
         "P&L": res_a.market["CCR default"] / MN},
        {"risk type": "CVA", "factor": "spread widening", "portfolio": "P&L", "P&L": res_a.market["CVA"] / MN}])
    return pd.concat([rv[["risk type", "factor", "portfolio", "sensitivity", "shock", "unit", "P&L"]], extra], ignore_index=True)


def nii(res_b, res_a) -> pd.DataFrame:
    return _long(res_b, res_a, lambda r: (r.nii / MN).reset_index())


def oprisk(res_b, res_a) -> pd.DataFrame:
    return _long(res_b, res_a, lambda r: (r.pnl[["operational losses", "conduct losses"]] / MN).reset_index())


def all_tables(res_b, res_a) -> dict[str, pd.DataFrame]:
    return {"CSV_SUM": summary(res_b, res_a), "CSV_CAP": capital(res_b, res_a), "CSV_P&L": pnl(res_b, res_a),
            "CSV_CR_SCEN": credit_params(res_b, res_a), "CSV_CR_PROV": credit_provisions(res_b, res_a), "CSV_CR_REA": credit_rea(res_b, res_a),
            "CSV_MR": market(res_a), "CSV_NII": nii(res_b, res_a), "CSV_OR": oprisk(res_b, res_a)}


def to_excel(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for k, v in tables.items():
            v.to_excel(xw, sheet_name=k.replace("&", "and")[:31], index=False)
    return buf.getvalue()
