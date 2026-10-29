"""
Net interest income projection (static balance sheet, constrained bottom-up).

Assets
  Loans: performing exposures earn  fixed share × locked rate + floating share × short rate_t  + client margin;
         stage-3 exposures earn interest on the net carrying amount (EAD − provisions) at the same rate.
         Asset margins cannot increase in the adverse scenario (no repricing of credit risk onto existing clients).
  Other assets (reserves, HQLA securities): β × short rate + spread, fixed share repricing gradually.
Liabilities
  Cost = fixed part locked + floating part at β × short rate_t + spread; wholesale funding (covered bonds, senior debt)
  reprices its share at the bank's stressed funding spread (adverse: + funding-spread shock).
  Deposit rates are floored at 0.
Constraint (optional, default on): adverse NII cannot exceed the starting-point NII in any year (EBA-style cap).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import scenario as SC


WHOLESALE = {"Covered bonds": 0.5, "Senior unsecured debt": 1.0, "Central bank & interbank funding": 0.5}


@dataclass
class NIIParams:
    funding_spread_shock: float = 0.0150     # adverse widening of the bank's wholesale funding spread
    cap_at_start: bool = True
    deposit_floor: float = 0.0


def project(bank, credit_proj: pd.DataFrame, path: pd.DataFrame, P: NIIParams, adverse: bool, scale: float = 1.0) -> pd.DataFrame:
    r0 = path["short rate"].iloc[0] / 100
    rows = []
    cr = bank.credit
    fixed_rate = r0          # locked rate on fixed-rate loans = starting short rate (simplification)
    for t in range(0, 4):
        r = path["short rate"].iloc[t] / 100
        cp = credit_proj[credit_proj["year"] == SC.START + t].set_index("row")
        perf = cp["S1"] + cp["S2"]
        net3 = (cp["S3"] - cp["prov S3"]).clip(lower=0)
        fx = cr["fixed_share"].reindex(cp.index).to_numpy()
        mg = cr["margin"].reindex(cp.index).to_numpy()
        yld = fx * fixed_rate + (1 - fx) * r + mg
        loans = float((perf.to_numpy() * yld).sum())
        stage3 = float((net3.to_numpy() * yld).sum())
        other_a = other_l = 0.0
        det = {}
        for f in bank.funding.to_dict("records"):
            rep = min(1.0, f["repricing"] * t)
            fixed_part = f["fixed share"] * (1 - rep)
            rate = fixed_part * (r0 * f["beta"] + f["spread"]) + (1 - fixed_part) * (r * f["beta"] + f["spread"])
            if adverse and f["item"] in WHOLESALE:
                rate += P.funding_spread_shock * scale * WHOLESALE[f["item"]] * rep
            if f["side"] == "liability":
                if "deposit" in f["item"].lower():
                    rate = max(rate, P.deposit_floor)
                other_l += f["amount"] * rate
                det[f["item"]] = -f["amount"] * rate
            else:
                other_a += f["amount"] * rate
                det[f["item"]] = f["amount"] * rate
        rows.append({"year": SC.START + t, "loans (performing)": loans, "loans (stage 3, net)": stage3, **det,
                     "interest income": loans + stage3 + other_a, "interest expense": other_l, "NII": loans + stage3 + other_a - other_l})
    df = pd.DataFrame(rows).set_index("year")
    df["NII uncapped"] = df["NII"]
    if adverse and P.cap_at_start:
        df.loc[df.index > SC.START, "NII"] = np.minimum(df.loc[df.index > SC.START, "NII"], df.loc[SC.START, "NII"])
    return df
