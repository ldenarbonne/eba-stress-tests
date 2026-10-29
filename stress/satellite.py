"""
Satellite models: link default rates to the macro variables of the scenario.

* A synthetic quarterly history (2005-2024) of EU macro variables — with the 2008-09 crisis, the sovereign debt crisis
  and the 2020 pandemic — and of 12-month default rates per portfolio, generated from a known model with
  autocorrelated errors (so that the estimation, its diagnostics and the backtest can be checked).
* Estimation per portfolio by OLS on the logit of the default rate:
  logit(DR_t) = a + Σ b_k x_k,t + e_t, with x in {GDP growth, Δ unemployment, HPI growth, CRE growth, short rate}
  (year-on-year quantities, so that annual scenario values can be plugged in directly).
* Diagnostics: Newey-West t-statistics, R², adjusted R², Durbin-Watson, VIF, expected-sign check, out-of-sample
  backtest on the last 12 quarters.
* Projection: PD shift in logit space, logit PD_t = logit PD_0 + (ŷ(x_t) − ŷ(x_0)) — the model moves the starting PD
  of each portfolio, it does not replace it (standard practice in EBA bottom-up projections).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

REGRESSORS = ["GDP growth", "Δ unemployment", "HPI growth", "CRE growth", "short rate"]
EXPECTED_SIGN = {"GDP growth": -1, "Δ unemployment": 1, "HPI growth": -1, "CRE growth": -1, "short rate": 1}

# true data-generating coefficients (on % units) and default specification per portfolio
TRUE = {
    "Corporates": {"const": -4.35, "GDP growth": -0.08, "Δ unemployment": 0.09, "short rate": 0.04},
    "SME": {"const": -3.85, "GDP growth": -0.07, "Δ unemployment": 0.12, "short rate": 0.05},
    "Commercial real estate": {"const": -4.05, "CRE growth": -0.045, "GDP growth": -0.05, "short rate": 0.06},
    "Residential mortgages": {"const": -4.80, "Δ unemployment": 0.13, "HPI growth": -0.022, "short rate": 0.04},
    "Consumer credit": {"const": -3.55, "Δ unemployment": 0.15, "GDP growth": -0.04},
    "Institutions": {"const": -6.40, "GDP growth": -0.12, "short rate": 0.03},
    "Sovereign": {"const": -6.70, "GDP growth": -0.10},
}
DEFAULT_SPEC = {p: [k for k in v if k != "const"] for p, v in TRUE.items()}


def history(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = pd.period_range("2005Q1", "2024Q4", freq="Q")
    n = len(q)
    t = np.arange(n)
    gdp = 1.6 + rng.normal(0, 0.5, n)
    gdp[(q >= pd.Period("2008Q4")) & (q <= pd.Period("2009Q4"))] = [-1.5, -4.5, -5.0, -4.2, -2.0]
    gdp[(q >= pd.Period("2012Q1")) & (q <= pd.Period("2013Q2"))] -= 2.2
    gdp[(q >= pd.Period("2020Q2")) & (q <= pd.Period("2021Q1"))] = [-14.0, -4.0, -4.5, -1.0]
    gdp[(q >= pd.Period("2021Q2")) & (q <= pd.Period("2021Q4"))] = [13.0, 4.0, 5.0]
    gdp = pd.Series(gdp).rolling(2, min_periods=1).mean().to_numpy()
    du = -0.35 * (gdp - 1.4) + rng.normal(0, 0.25, n)
    du[(q >= pd.Period("2020Q2")) & (q <= pd.Period("2021Q4"))] *= 0.25       # job-retention schemes
    unemp = 8.5 + np.cumsum(du) / 4
    unemp = np.clip(unemp, 6.0, 12.5)
    d_unemp = np.r_[np.zeros(4), unemp[4:] - unemp[:-4]]
    hpi = 4.0 + 1.6 * (gdp - 1.4) + rng.normal(0, 1.5, n)
    hpi[(q >= pd.Period("2022Q3"))] -= 4.0
    cre = 2.0 + 2.2 * (gdp - 1.4) + rng.normal(0, 2.5, n)
    cre[(q >= pd.Period("2022Q3"))] -= 9.0
    rate = np.interp(t, [0, 12, 16, 20, 28, 40, 60, 68, 72, 76, 79], [2.2, 4.2, 4.5, 1.0, 0.8, 0.1, -0.5, -0.5, 1.5, 3.8, 3.0])
    X = pd.DataFrame({"GDP growth": gdp, "Δ unemployment": d_unemp, "HPI growth": hpi, "CRE growth": cre, "short rate": rate},
                     index=q.to_timestamp(how="end").normalize())
    for p, coef in TRUE.items():
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.65 * e[i - 1] + rng.normal(0, 0.07)
        y = coef["const"] + sum(b * X[k] for k, b in coef.items() if k != "const") + e
        X[f"DR {p}"] = expit(y)
    return X


@dataclass
class Fit:
    portfolio: str
    variables: list
    coef: pd.Series
    se_nw: pd.Series
    r2: float
    r2_adj: float
    dw: float
    vif: pd.Series
    fitted: pd.Series
    actual: pd.Series
    oos: pd.DataFrame
    rmse_in: float
    rmse_oos: float

    @property
    def t(self):
        return self.coef / self.se_nw

    def sign_ok(self) -> pd.Series:
        return pd.Series({k: np.sign(self.coef[k]) == EXPECTED_SIGN[k] for k in self.variables})

    def predict_logit(self, x: pd.DataFrame | pd.Series) -> np.ndarray:
        x = x if isinstance(x, pd.DataFrame) else x.to_frame().T
        return self.coef["const"] + (x[self.variables].to_numpy() @ self.coef[self.variables].to_numpy())


def _ols(y, X, lags=4):
    Xc = np.column_stack([np.ones(len(y)), X])
    inv = np.linalg.inv(Xc.T @ Xc)
    b = inv @ Xc.T @ y
    e = y - Xc @ b
    Xe = Xc * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        G = Xe[l:].T @ Xe[:-l]
        S += (1 - l / (lags + 1)) * (G + G.T)
    V = inv @ S @ inv * len(y) / (len(y) - Xc.shape[1])
    return b, np.sqrt(np.diag(V)), e


def fit(hist: pd.DataFrame, portfolio: str, variables: list[str], holdout: int = 12) -> Fit:
    df = hist.iloc[4:]                            # first year has no y-o-y change of unemployment
    y = logit(df[f"DR {portfolio}"].to_numpy())
    X = df[variables].to_numpy()
    tr = slice(0, len(y) - holdout)
    b, se, e = _ols(y[tr], X[tr])
    names = ["const"] + list(variables)
    coef, se = pd.Series(b, names), pd.Series(se, names)
    yhat_all = b[0] + X @ b[1:]
    tss = ((y[tr] - y[tr].mean()) ** 2).sum()
    r2 = 1 - (e @ e) / tss
    n, k = len(e), len(names)
    dw = float((np.diff(e) ** 2).sum() / (e @ e))
    vif = {}
    for i, v in enumerate(variables):
        if len(variables) == 1:
            vif[v] = 1.0
            continue
        others = np.delete(X[tr], i, axis=1)
        _, _, ei = _ols(X[tr][:, i], others, 0)
        r2i = 1 - (ei @ ei) / ((X[tr][:, i] - X[tr][:, i].mean()) ** 2).sum()
        vif[v] = 1 / max(1 - r2i, 1e-9)
    idx = df.index
    oos = pd.DataFrame({"actual": expit(y[len(y) - holdout:]), "predicted": expit(yhat_all[len(y) - holdout:])}, index=idx[len(y) - holdout:])
    return Fit(portfolio, list(variables), coef, se, float(r2), float(1 - (1 - r2) * (n - 1) / (n - k)), dw, pd.Series(vif),
               pd.Series(expit(yhat_all), idx), pd.Series(expit(y), idx), oos,
               float(np.sqrt(np.mean((expit(yhat_all[tr]) - expit(y[tr])) ** 2))),
               float(np.sqrt(np.mean((oos["predicted"] - oos["actual"]) ** 2))))


def fit_all(hist: pd.DataFrame, spec: dict | None = None) -> dict[str, Fit]:
    spec = spec or DEFAULT_SPEC
    return {p: fit(hist, p, v) for p, v in spec.items()}


def pd_path(fit_: Fit, pd0: float, x_start: pd.Series, x_path: pd.DataFrame, floor: float = 0.0003) -> np.ndarray:
    """Projected 12-month PD per year (logit shift)."""
    shift = fit_.predict_logit(x_path) - fit_.predict_logit(x_start)[0]
    return np.maximum(expit(logit(np.clip(pd0, 1e-6, 1 - 1e-6)) + shift), floor)
