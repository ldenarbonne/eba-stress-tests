"""
Credit risk projection (bottom-up, static balance sheet, annual steps).

For each portfolio × country row and each year t = 1..3:

Stage transitions (shares of the opening stock)
  S1→S3 = PD_t                         S1→S2 = min(m_SICR · PD_t, 35 %)
  S2→S3 = min(m_S2 · PD_t, 60 %)       S2→S1 = cure_0 · PD_0 / PD_t   (cures fall when risk rises)
  S3→S1/S2 = 0 by default              (no cure of defaulted exposures — conservative EBA-style constraint)
  Defaulted exposures are not replaced (static balance sheet): the performing book shrinks by new defaults.

Loss parameters
  PD_t from the satellite model (logit shift of the starting PD).
  LGD_t: secured portfolios (residential mortgages, CRE) by a collateral model driven by the house / CRE price index,
         LGD = 1 − (1 − haircut) · index / LTV_default + workout cost (moved from its starting level);
         unsecured: LGD_0 · (1 + λ · Δ unemployment).
  Stage-3 coverage: new defaults at LGD_t; coverage of the existing stock cannot decrease.

Provisions (IFRS 9 stock)          S1: PD_t · LGD_t · EAD          (12-month ECL)
                                   S2: [1 − (1 − PD_S2,t)(1 − PD_t)^(min(M, 5) − 1)] · LGD_t · EAD  (lifetime ECL)
                                   S3: coverage_t · EAD
  Impairment charge_t = provisions_t − provisions_{t−1}.

REA
  SA: risk weight × (exposure − provisions); defaulted: 100 % if specific provisions ≥ 20 %, otherwise 150 %.
  IRB: CRR risk-weight function with PD_reg,t = PD_reg,0 · (PD_t / PD_0)^ε (rating migration of the regulatory PD,
       ε = cyclicality), S2 exposures at m_S2 × PD_reg, downturn LGD constant; defaulted: RW = 12.5 × (LGD_DT − ELBE).
  IRB shortfall (EL − provisions on performing IRB exposures) is deducted from CET1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import satellite as SAT
from . import scenario as SC

SECURED = {"Residential mortgages": ("HPI growth", 0.25, 0.90, 0.05), "Commercial real estate": ("CRE growth", 0.35, 0.80, 0.05)}


@dataclass
class CreditParams:
    m_sicr: float = 3.0          # S1→S2 transfer rate as a multiple of PD
    m_s2: float = 3.5            # PD of stage-2 exposures as a multiple of PD
    cure0: float = 0.22          # S2→S1 cure rate at the starting point
    s3_cure: float = 0.0         # S3→S2 cure rate (0 = not allowed)
    lgd_lambda: float = 0.04     # unsecured LGD sensitivity per pp of unemployment increase
    elasticity: float = 0.7      # regulatory PD response to PIT PD (0 = pure TTC, 1 = PIT)
    lgd_dt_addon: float = 0.03   # downturn LGD add-on for IRB REA
    lifetime_cap: float = 5.0
    rea_floor_start: bool = True # credit REA cannot fall below the starting point


def irb_rw(pd_, lgd, M, cls: str) -> np.ndarray:
    pd_ = np.maximum(np.asarray(pd_, float), 0.0005 if cls != "mortgage" else 0.001)
    if cls == "mortgage":
        R = np.full_like(pd_, 0.15)
        ma = 1.0
    else:
        w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
        R = 0.12 * w + 0.24 * (1 - w)
        if cls == "sme":
            R = R - 0.04 * (1 - (25 - 5) / 45)
        if cls == "institution":
            R = R * 1.25
        b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
        ma = (1 + (np.clip(M, 1, 5) - 2.5) * b) / (1 - 1.5 * b)
    k = lgd * (norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_) * ma
    return 12.5 * k


def _secured_lgd(lgd0, idx, haircut, ltv, cost):
    struct = lambda i: max(0.0, 1 - (1 - haircut) * i / ltv) + cost  # noqa: E731
    return float(np.clip(lgd0 + struct(idx) - struct(1.0), 0.05, 0.9))


def project(credit: pd.DataFrame, paths: dict, fits: dict, P: CreditParams, scenario: str = "adverse", scale: float = 1.0,
            country_mult: pd.Series | None = None) -> pd.DataFrame:
    """Returns one row per (row id, year) with stages, flows, provisions, impairments, REA (year 0 = starting point)."""
    out = []
    for rid, r in credit.iterrows():
        base = paths["baseline"]
        adv = paths["adverse"] if scenario == "adverse" else paths["baseline"]
        path = SC.country_path(base, adv, r["country"], scale if scenario == "adverse" else 0.0, country_mult)
        x0 = SC.start_regressors(path)
        X = SC.regressors(path)
        f = fits.get(r["portfolio"])
        pds = SAT.pd_path(f, r["pd"], x0, X) if f is not None else np.full(3, r["pd"])
        pd_s = np.r_[r["pd"], pds]
        s1, s2, s3 = r["ead_s1"], r["ead_s2"], r["ead_s3"]
        cov = r["s3_coverage"]
        lgd0 = r["lgd"]
        u0 = path["unemployment"].iloc[0]
        idx = 1.0
        prev_lgd = lgd0
        prov_prev = None
        for t in range(0, 4):
            p = pd_s[t]
            if t == 0:
                lgd = lgd0
                new_def = tr12 = cure = s3cure = 0.0
            else:
                if r["portfolio"] in SECURED:
                    var, hc, ltv, cost = SECURED[r["portfolio"]]
                    idx *= 1 + path[var].iloc[t] / 100
                    lgd = _secured_lgd(lgd0, idx, hc, ltv, cost)
                else:
                    lgd = float(np.clip(lgd0 * (1 + P.lgd_lambda * max(0.0, path["unemployment"].iloc[t] - u0)), 0.05, 0.95))
                d13 = min(p, 1.0)
                d12 = min(P.m_sicr * p, 0.35)
                d23 = min(P.m_s2 * p, 0.6)
                d21 = min(P.cure0 * r["pd"] / p, 0.5) if p > 0 else 0
                f13, f12, f23, f21 = s1 * d13, s1 * d12, s2 * d23, s2 * d21
                s3c = s3 * P.s3_cure
                new_def = f13 + f23
                tr12 = f12
                cure = f21
                s3cure = s3c
                old_s3 = s3
                s1 = s1 - f13 - f12 + f21
                s2 = s2 - f23 - f21 + f12 + s3c
                s3 = s3 + new_def - s3c
                cov_old = max(cov, cov + (lgd - prev_lgd))
                cov = (cov_old * (old_s3 - s3c) + lgd * new_def) / max(s3, 1e-9)
                prev_lgd = lgd
            pd2 = min(P.m_s2 * p, 0.6)
            life = 1 - (1 - pd2) * (1 - p) ** (min(r["maturity"], P.lifetime_cap) - 1)
            prov1, prov2, prov3 = s1 * p * lgd, s2 * life * lgd, s3 * cov
            prov = prov1 + prov2 + prov3
            imp = 0.0 if prov_prev is None else prov - prov_prev
            prov_prev = prov
            # REA
            if r["approach"] == "SA":
                rea_perf = r["rw_sa"] * max(s1 + s2 - prov1 - prov2, 0)
                rea_def = (1.0 if cov >= 0.2 else 1.5) * max(s3 - prov3, 0)
                el_irb = prov_irb = 0.0
                pdr = r["pd_reg"]
            else:
                pdr = r["pd_reg"] * (p / r["pd"]) ** P.elasticity
                lgd_reg = min(lgd0 + P.lgd_dt_addon, 0.95)
                rea_perf = s1 * irb_rw(pdr, lgd_reg, r["maturity"], r["irb_class"]) + s2 * irb_rw(min(P.m_s2 * pdr, 0.6), lgd_reg, r["maturity"], r["irb_class"])
                rea_def = 12.5 * P.lgd_dt_addon * s3 * 1.6           # LGD_DT − ELBE ≈ 8 % → RW ≈ 100 %
                rea_perf, rea_def = float(np.squeeze(rea_perf)), float(rea_def)
                el_irb = s1 * pdr * lgd_reg + s2 * min(P.m_s2 * pdr, 0.6) * lgd_reg
                prov_irb = prov1 + prov2
            sa_eq = r["rw_sa"] * max(s1 + s2 - prov1 - prov2, 0) + (1.0 if cov >= 0.2 else 1.5) * max(s3 - prov3, 0)
            out.append({"row": rid, "portfolio": r["portfolio"], "country": r["country"], "approach": r["approach"], "year": SC.START + t,
                        "S1": s1, "S2": s2, "S3": s3, "EAD": s1 + s2 + s3, "PD": p, "PD reg": pdr, "LGD": lgd, "S3 coverage": cov,
                        "new defaults": new_def, "S1→S2": tr12, "S2→S1": cure, "S3 cures": s3cure,
                        "prov S1": prov1, "prov S2": prov2, "prov S3": prov3, "provisions": prov, "impairment": imp,
                        "REA": rea_perf + rea_def, "REA SA-equivalent": sa_eq, "EL IRB": el_irb, "prov IRB perf": prov_irb})
    return pd.DataFrame(out)


def summary(proj: pd.DataFrame, by=("year",)) -> pd.DataFrame:
    g = proj.groupby(list(by))[["S1", "S2", "S3", "EAD", "new defaults", "provisions", "impairment", "REA", "REA SA-equivalent"]].sum()
    g["S3 ratio"] = g["S3"] / g["EAD"]
    g["coverage S3"] = proj.groupby(list(by))["prov S3"].sum() / g["S3"]
    g["default rate"] = g["new defaults"] / (g["S1"] + g["S2"] + g["new defaults"])
    return g
