"""EBA-style EU-wide stress test – credit, market & CCR, NII, operational & conduct risk, capital and leverage."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stress import credit as CR
from stress import data as D
from stress import engine as E
from stress import market as MK
from stress import nii as NI
from stress import satellite as SAT
from stress import scenario as SC
from stress import templates as TP

st.set_page_config(page_title="EBA Stress Test", page_icon="🏦", layout="wide")
NAVY, RED, GREY, ORANGE, GREEN, TEAL = "#1d3557", "#c1121f", "#9aa5b1", "#e09f3e", "#2a9d8f", "#457b9d"
LEG = dict(orientation="h", y=-0.2)
BN, MN = 1e9, 1e6


def h(df):
    return min(35 * (len(df) + 1) + 3, 900)


def text(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    out = df.copy()
    out.attrs = {}
    for c in out.columns:
        f = fmt.get(c)
        if f is not None:
            out[c] = [("–" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f.format(v)) for v in df[c]]
    return out


def key_of(*objs) -> str:
    m = hashlib.md5()
    for o in objs:
        m.update((o.to_json() if isinstance(o, (pd.DataFrame, pd.Series)) else repr(o)).encode())
    return m.hexdigest()


@st.cache_data(show_spinner=False)
def history(seed):
    return SAT.history(seed)


@st.cache_data(show_spinner="Running the stress test…")
def run_both(_bank, _paths, _shocks, _fits, _P, key):
    return E.run(_bank, _paths, _shocks, _fits, _P, "baseline"), E.run(_bank, _paths, _shocks, _fits, _P, "adverse")


@st.cache_data(show_spinner="One-at-a-time contributions…")
def run_contrib(_bank, _paths, _shocks, _fits, _P, key):
    return E.contributions(_bank, _paths, _shocks, _fits, _P)


@st.cache_data(show_spinner="Sensitivity analysis…")
def run_tornado(_bank, _paths, _shocks, _fits, _P, key):
    return E.tornado(_bank, _paths, _shocks, _fits, _P)


@st.cache_data(show_spinner="Reverse stress test…")
def run_reverse(_bank, _paths, _shocks, _fits, _P, key, thr):
    return E.reverse_stress(_bank, _paths, _shocks, _fits, _P, thr)


# ---------------------------------------------------------------- sidebar: methodology parameters
with st.sidebar:
    st.header("Credit risk")
    m_sicr = st.slider("S1→S2 transfer rate (× PD)", 1.0, 8.0, 3.0, 0.5)
    m_s2 = st.slider("Stage-2 PD (× PD)", 1.5, 6.0, 3.5, 0.25)
    cure0 = st.slider("S2→S1 cure rate at start", 0.0, 0.5, 0.22, 0.01)
    s3_cure = st.slider("S3 cure rate (EBA-style: 0)", 0.0, 0.2, 0.0, 0.01)
    lgd_l = st.slider("Unsecured LGD sensitivity (per pp unemployment)", 0.0, 0.10, 0.04, 0.005)
    elast = st.slider("Regulatory PD cyclicality (0 = TTC, 1 = PIT)", 0.0, 1.0, 0.7, 0.05)
    rea_floor = st.checkbox("Credit REA floored at the starting point", value=True)
    st.header("Market & CCR")
    ava_m = st.slider("AVA increase (× starting AVA)", 0.0, 2.0, 0.5, 0.1)
    n_def = st.select_slider("Counterparty defaults", [0, 1, 2, 3], value=2)
    tr_hc = st.slider("Client trading revenue haircut (Y2-Y3)", 0.0, 1.0, 0.5, 0.05)
    st.header("NII")
    fs = st.slider("Wholesale funding spread shock (bp)", 0, 400, 150, 10) / 1e4
    cap_nii = st.checkbox("Adverse NII capped at the starting point", value=True)
    st.header("Other P&L")
    fee_hc = st.slider("Fee & commission haircut (adverse)", 0.0, 0.4, 0.12, 0.01)
    op_m = st.slider("Operational losses (× historical average)", 1.0, 3.0, 1.6, 0.1)
    cd_m = st.slider("Non-material conduct losses (× historical)", 1.0, 3.0, 1.3, 0.1)
    infl = st.checkbox("Expenses indexed on scenario inflation (floored at start)", value=True)
    dta = st.checkbox("Tax credit on losses (DTA recognised)", value=False)
    ofl = st.checkbox("Apply the CRR3 output floor", value=True)

P = E.Params(credit=CR.CreditParams(m_sicr=m_sicr, m_s2=m_s2, cure0=cure0, s3_cure=s3_cure, lgd_lambda=lgd_l, elasticity=elast,
                                     rea_floor_start=rea_floor),
             market=MK.MarketParams(ava_multiplier=ava_m, n_defaults=n_def, trading_haircut=tr_hc),
             nii=NI.NIIParams(funding_spread_shock=fs, cap_at_start=cap_nii), fee_haircut=fee_hc, op_multiplier=op_m,
             conduct_multiplier=cd_m, expenses_inflation=infl, dta_on_losses=dta, output_floor=ofl)

# ---------------------------------------------------------------- data
bank = D.generate()
if "credit_import" in st.session_state:
    bank.credit = st.session_state["credit_import"]

st.title("EU-wide Stress Test Engine")
st.caption("Constrained bottom-up stress test of a mid-sized EU bank in the spirit of the EBA methodology: three-year horizon "
           "(2025-2027), static balance sheet, baseline and adverse scenarios — credit risk with IFRS 9 stages and REA, market "
           "risk and counterparty defaults, net interest income, operational and conduct risk, CET1 and leverage ratios.")

tabs = st.tabs(["Summary", "Scenarios", "Satellite models", "Credit risk", "Market & CCR", "NII", "P&L", "Capital & leverage",
                "Sensitivity & reverse", "Templates", "Data & import", "Methodology"])

# scenario editors first (they feed everything)
with tabs[1]:
    st.markdown("Annual paths for the euro-area aggregate; 2024 is the starting point. The adverse scenario is applied to each "
                "country with a severity multiplier on its deviation from the baseline.")
    dp = SC.default_paths()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline**")
        base_ed = st.data_editor(dp["baseline"], key="base_ed", width="stretch")
    with c2:
        st.markdown("**Adverse**")
        adv_ed = st.data_editor(dp["adverse"], key="adv_ed", width="stretch")
    paths = {"baseline": base_ed.astype(float), "adverse": adv_ed.astype(float)}
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Country severity multipliers**")
        cm = st.data_editor(SC.COUNTRY_MULT.rename("multiplier").to_frame(), key="cm_ed", width="stretch")
        P.country_mult = cm["multiplier"].astype(float)
    with c2:
        st.markdown("**Instantaneous market shocks (adverse)**")
        shocks = st.data_editor(SC.default_market_shocks(), key="shock_ed", width="stretch", hide_index=True)
    k = st.columns(4)
    gdp_cum = (np.prod(1 + paths["adverse"]["GDP growth"].iloc[1:] / 100) - 1)
    k[0].metric("Adverse GDP, cumulative 2025-27", f"{gdp_cum:+.1%}")
    k[1].metric("Unemployment gap vs baseline, 2027", f"{paths['adverse']['unemployment'].iloc[-1] - paths['baseline']['unemployment'].iloc[-1]:+.1f} pp")
    k[2].metric("House prices, cumulative", f"{SC.cumulative(paths['adverse'], 'HPI growth').iloc[-1] / 100 - 1:+.1%}")
    k[3].metric("Commercial real estate, cumulative", f"{SC.cumulative(paths['adverse'], 'CRE growth').iloc[-1] / 100 - 1:+.1%}")
    cols = st.columns(4)
    for i, (v, lab) in enumerate([("GDP growth", "Real GDP growth (%)"), ("unemployment", "Unemployment rate (%)"), ("HPI growth", "House price growth (%)"),
                                  ("short rate", "Short-term rate (%)")]):
        fig = go.Figure()
        fig.add_scatter(x=paths["baseline"].index, y=paths["baseline"][v], name="baseline", line_color=NAVY)
        fig.add_scatter(x=paths["adverse"].index, y=paths["adverse"][v], name="adverse", line_color=RED)
        fig.update_layout(height=260, margin=dict(t=30, b=10), title=lab, showlegend=i == 0, legend=LEG, xaxis=dict(dtick=1))
        cols[i].plotly_chart(fig, width="stretch")
    st.caption("The default adverse scenario is illustrative: calibrated on the headline figures of the ESRB adverse scenario of "
               "the 2025 EU-wide stress test (EU GDP about −6.3 % cumulatively over 2025-2027, unemployment about 6 pp above "
               "baseline, inflation 5.0 / 3.5 / 1.9 %); the other variables and the market shocks are plausible values for the demo.")

hist = history(9)
with tabs[2]:
    st.markdown("Default-rate models estimated on a synthetic quarterly history (2005-2024): logit(DR) regressed on year-on-year "
                "macro variables. The projection shifts each portfolio's starting PD by the model's predicted change.")
    port = st.selectbox("Portfolio", list(SAT.DEFAULT_SPEC))
    spec = {}
    for p_ in SAT.DEFAULT_SPEC:
        spec[p_] = st.session_state.get(f"spec_{p_}", SAT.DEFAULT_SPEC[p_])
    sel = st.multiselect("Explanatory variables", SAT.REGRESSORS, default=SAT.DEFAULT_SPEC[port], key=f"spec_{port}")
    if not sel:
        st.warning("Select at least one variable.")
        sel = SAT.DEFAULT_SPEC[port]
    spec[port] = sel
fits = SAT.fit_all(hist, spec)
with tabs[2]:
    f = fits[port]
    k = st.columns(5)
    k[0].metric("R² (in-sample)", f"{f.r2:.2f}")
    k[1].metric("Adjusted R²", f"{f.r2_adj:.2f}")
    k[2].metric("Durbin-Watson", f"{f.dw:.2f}", help="≈ 2: no autocorrelation; < 1.5: positive autocorrelation (t-stats use Newey-West).")
    k[3].metric("RMSE in-sample", f"{f.rmse_in * 1e4:.0f} bp")
    k[4].metric("RMSE out-of-sample (12 q)", f"{f.rmse_oos * 1e4:.0f} bp")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=f.actual.index, y=f.actual, name="observed default rate", line_color=NAVY)
        fig.add_scatter(x=f.fitted.index, y=f.fitted, name="model", line=dict(color=ORANGE, dash="dash"))
        fig.add_vrect(x0=f.oos.index[0], x1=f.oos.index[-1], fillcolor=GREY, opacity=0.2, line_width=0, annotation_text="out-of-sample")
        fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_tickformat=".1%", legend=LEG, title=f"{port}: 12-month default rate")
        st.plotly_chart(fig, width="stretch")
    with c2:
        t = pd.DataFrame({"coefficient": f.coef, "t (Newey-West)": f.t, "VIF": f.vif.reindex(f.coef.index)})
        t["expected sign"] = ["" if i == "const" else ("+" if SAT.EXPECTED_SIGN[i] > 0 else "−") for i in t.index]
        t["sign OK"] = ["" if i == "const" else ("✓" if f.sign_ok()[i] else "✗") for i in t.index]
        st.dataframe(text(t, {"coefficient": "{:.4f}", "t (Newey-West)": "{:.2f}", "VIF": "{:.2f}"}), width="stretch")
        st.caption("Validation checklist: economically intuitive signs, significance with HAC errors, VIF < 5 (multicollinearity), "
                   "stable out-of-sample error.")
    st.subheader("Projected PD multipliers (euro-area aggregate path)")
    x0b = SC.start_regressors(paths["baseline"])
    rows = []
    for p_, ff in fits.items():
        for scn in ["baseline", "adverse"]:
            pdp = SAT.pd_path(ff, 0.01, x0b, SC.regressors(paths[scn]))
            for y, v in zip(SC.YEARS, pdp):
                rows.append({"portfolio": p_, "scenario": scn, "year": y, "PD multiplier": v / 0.01})
    pm = pd.DataFrame(rows)
    pm["year"] = pm["year"].astype(str)
    fig = px.bar(pm[pm["scenario"] == "adverse"], x="portfolio", y="PD multiplier", color="year", barmode="group",
                 color_continuous_scale=None, color_discrete_sequence=[ORANGE, RED, NAVY])
    fig.add_hline(y=1, line_dash="dot", line_color=GREY)
    fig.update_layout(height=360, margin=dict(t=30, b=10), legend=LEG, title="Adverse PD relative to the starting PD (1 % reference)")
    st.plotly_chart(fig, width="stretch")

key = key_of(paths["baseline"], paths["adverse"], shocks, P.country_mult, repr(P), bank.credit, repr(spec))
res_b, res_a = run_both(bank, paths, shocks, fits, P, key)
FL, TR = "CET1 ratio (fully loaded)", "CET1 ratio (transitional)"

# ---------------------------------------------------------------- summary
with tabs[0]:
    ca, cb = res_a.capital, res_b.capital
    k = st.columns(6)
    k[0].metric("CET1 ratio, start (fully loaded)", f"{ca[FL].iloc[0]:.2%}")
    k[1].metric("Adverse, end 2027", f"{ca[FL].iloc[-1]:.2%}", f"{(ca[FL].iloc[-1] - ca[FL].iloc[0]) * 1e4:+.0f} bp")
    k[2].metric("Adverse, lowest point", f"{ca[FL].min():.2%}", f"max depletion {(ca[FL].iloc[0] - ca[FL].min()) * 1e4:.0f} bp", delta_color="off")
    k[3].metric("Baseline, end 2027", f"{cb[FL].iloc[-1]:.2%}", f"{(cb[FL].iloc[-1] - cb[FL].iloc[0]) * 1e4:+.0f} bp")
    k[4].metric("Leverage ratio, adverse end", f"{ca['leverage ratio'].iloc[-1]:.2%}", f"start {ca['leverage ratio'].iloc[0]:.2%}", delta_color="off")
    k[5].metric("Adverse cumulative net result", f"€{res_a.pnl['net profit'].sum() / BN:+.2f}bn")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for r, col in ((res_b, NAVY), (res_a, RED)):
            fig.add_scatter(x=r.capital.index, y=r.capital[FL], name=f"{r.scenario} — fully loaded", line=dict(color=col, width=3))
            fig.add_scatter(x=r.capital.index, y=r.capital[TR], name=f"{r.scenario} — transitional", line=dict(color=col, dash="dot"))
        fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_tickformat=".1%", legend=LEG, xaxis=dict(dtick=1), title="CET1 ratio")
        st.plotly_chart(fig, width="stretch")
    with c2:
        w = res_a.waterfall
        fig = go.Figure(go.Waterfall(x=["start"] + list(w.index) + ["end 2027"], y=[w.attrs["start"] * 1e4] + list(w.values * 1e4) + [0],
                                     measure=["absolute"] + ["relative"] * len(w) + ["total"],
                                     text=[f"{w.attrs['start']:.1%}"] + [f"{v * 1e4:+.0f}" for v in w.values] + [f"{w.attrs['end']:.1%}"],
                                     increasing_marker_color=GREEN, decreasing_marker_color=RED, totals_marker_color=NAVY))
        fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_title="bp of REA", title="Adverse: CET1 ratio change by driver (fully loaded)",
                          yaxis_range=[min(w.attrs["low"], w.attrs["end"]) * 1e4 - 600, None])
        st.plotly_chart(fig, width="stretch")
    st.dataframe(text(TP.summary(res_b, res_a), {"start": "{:.2%}", "end": "{:.2%}", "lowest": "{:.2%}", "change start→end (bp)": "{:+.0f}",
                                                "max depletion (bp)": "{:.0f}"}), width="stretch", hide_index=True)
    gross = pd.Series({"credit risk (impairments)": -res_a.pnl["impairments"].sum(),
                       "market risk & CCR (incl. OCI and AVA)": -(res_a.pnl["market shock (HFT, CCR, CVA)"].sum() + res_a.capital["OCI (FVOCI)"].sum() + res_a.capital["AVA"].sum()),
                       "operational & conduct": -(res_a.pnl["operational losses"].sum() + res_a.pnl["conduct losses"].sum())})
    rea0 = res_a.capital["REA fully loaded"].iloc[0]
    st.markdown("**Gross adverse losses and their weight in CET1-ratio points** — for comparison, the 2025 EU-wide aggregate "
                "reported 437 bp (credit), 108 bp (market) and 61 bp (operational) on a 370 bp total depletion.")
    gt = pd.DataFrame({"loss (€bn)": gross / BN, "bp of starting REA": gross / rea0 * 1e4})
    st.dataframe(text(gt, {"loss (€bn)": "{:.2f}", "bp of starting REA": "{:.0f}"}), width="stretch")

# ---------------------------------------------------------------- credit
with tabs[3]:
    ccp = {s: CR.summary(r.credit) for s, r in (("baseline", res_b), ("adverse", res_a))}
    k = st.columns(4)
    k[0].metric("Adverse impairments 2025-27", f"€{-res_a.pnl['impairments'].sum() / BN:.2f}bn",
                f"baseline €{-res_b.pnl['impairments'].sum() / BN:.2f}bn", delta_color="off")
    k[1].metric("Cumulative loss rate (adverse)", f"{-res_a.pnl['impairments'].sum() / ccp['adverse']['EAD'].iloc[0]:.2%}")
    k[2].metric("Stage-3 ratio, end 2027", f"{ccp['adverse']['S3 ratio'].iloc[-1]:.1%}", f"start {ccp['adverse']['S3 ratio'].iloc[0]:.1%}", delta_color="off")
    k[3].metric("Credit REA, end vs start", f"{res_a.capital['credit REA'].iloc[-1] / res_a.capital['credit REA'].iloc[0] - 1:+.1%}")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for s_, col in (("S1", GREEN), ("S2", ORANGE), ("S3", RED)):
            fig.add_bar(x=[f"{y} B" for y in ccp["baseline"].index], y=ccp["baseline"][s_] / BN, name=s_, marker_color=col, legendgroup=s_)
            fig.add_bar(x=[f"{y} A" for y in ccp["adverse"].index], y=ccp["adverse"][s_] / BN, name=s_, marker_color=col, legendgroup=s_, showlegend=False)
        fig.update_layout(barmode="stack", height=400, margin=dict(t=30, b=10), yaxis_title="€bn", legend=LEG,
                          title="Exposure by IFRS 9 stage (B = baseline, A = adverse)", yaxis_range=[150, None])
        st.plotly_chart(fig, width="stretch")
    with c2:
        imp = res_a.credit[res_a.credit["year"] > SC.START].groupby(["portfolio", "year"])["impairment"].sum().unstack() / MN
        fig = go.Figure()
        for i, y in enumerate(SC.YEARS):
            fig.add_bar(y=imp.index, x=imp[y], name=str(y), orientation="h", marker_color=[ORANGE, RED, NAVY][i])
        fig.update_layout(barmode="stack", height=400, margin=dict(t=30, b=10), xaxis_title="€m", legend=LEG, title="Adverse impairments by portfolio")
        st.plotly_chart(fig, width="stretch")
    cp_t = TP.credit_params(res_b, res_a)
    cp_t = cp_t[cp_t["year"] > SC.START]
    view = cp_t.pivot_table(index="portfolio", columns=["scenario", "year"], values="default rate")
    st.markdown("**Default rates (new defaults / opening performing exposure)**")
    st.dataframe(view.map(lambda v: f"{v:.2%}"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        lr = res_a.credit.groupby(["portfolio", "country"]).apply(lambda g: g["impairment"].sum() / g.loc[g["year"] == SC.START, "EAD"].sum(),
                                                                   include_groups=False).unstack()
        fig = px.imshow(lr, text_auto=".1%", color_continuous_scale="Reds", aspect="auto")
        fig.update_layout(height=380, margin=dict(t=30, b=10), title="Adverse cumulative loss rate by portfolio and country", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        rea = res_a.credit.groupby(["year", "approach"])["REA"].sum().unstack() / BN
        fig = go.Figure()
        for ap, col in (("IRB", NAVY), ("SA", TEAL)):
            fig.add_bar(x=rea.index, y=rea[ap], name=ap, marker_color=col)
        fig.add_scatter(x=res_a.capital.index, y=res_a.capital["credit REA"] / BN, name="credit REA used (floored at start)", mode="lines+markers", line_color=RED)
        fig.update_layout(barmode="stack", height=380, margin=dict(t=30, b=10), yaxis_title="€bn", legend=LEG, title="Adverse credit REA",
                          xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Stage flows, provisions and coverage (adverse, € million)**")
    pv = TP.credit_provisions(res_b, res_a)
    pv = pv[pv["scenario"] == "adverse"].drop(columns="scenario")
    st.dataframe(text(pv, {c: "{:,.0f}" for c in pv.columns if c not in ("portfolio", "year")}), width="stretch", hide_index=True, height=360)

# ---------------------------------------------------------------- market
with tabs[4]:
    mk = res_a.market
    k = st.columns(5)
    k[0].metric("HFT revaluation (P&L)", f"€{mk['HFT'] / MN:,.0f}m")
    k[1].metric("FVOCI revaluation (OCI)", f"€{mk['FVOCI'] / MN:,.0f}m")
    k[2].metric("AVA increase (CET1 deduction)", f"€{mk['AVA'] / MN:,.0f}m")
    k[3].metric("Counterparty defaults", f"€{mk['CCR default'] / MN:,.0f}m")
    k[4].metric("CVA", f"€{mk['CVA'] / MN:,.0f}m")
    c1, c2 = st.columns([3, 2])
    with c1:
        rv = mk["revaluation"].assign(label=lambda d: d["risk type"] + " – " + d["factor"])
        fig = px.bar(rv.sort_values("P&L"), x="P&L", y="label", color="portfolio", orientation="h",
                     color_discrete_map={"HFT": NAVY, "FVOCI": ORANGE})
        fig.update_layout(height=520, margin=dict(t=30, b=10), xaxis_title="€", yaxis_title=None, legend=LEG, title="Instantaneous revaluation by risk factor")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Defaulting counterparties** (two most vulnerable among the ten largest, excluding sovereigns and CCPs)")
        d_ = mk["defaulted"]
        st.dataframe(text(d_[["counterparty", "type", "rating", "stressed EAD", "collateral", "LGD", "loss"]],
                          {"stressed EAD": "€{:,.0f}", "collateral": "€{:,.0f}", "LGD": "{:.0%}", "loss": "€{:,.0f}"}), width="stretch", hide_index=True)
        st.markdown("**Top-10 counterparties**")
        st.dataframe(text(bank.cpty, {"stressed EAD": "€{:,.0f}", "collateral": "€{:,.0f}", "LGD": "{:.0%}"}), width="stretch", hide_index=True, height=h(bank.cpty))
    st.markdown("**Trading income** (adverse: no client revenue in year 1, haircut in years 2-3)")
    ti = pd.DataFrame({"baseline": res_b.pnl["trading income"] / MN, "adverse": res_a.pnl["trading income"] / MN}).T
    st.dataframe(ti.map(lambda v: f"{v:,.0f}"), width="stretch")

# ---------------------------------------------------------------- NII
with tabs[5]:
    k = st.columns(4)
    n0 = res_a.nii["NII"].iloc[0]
    k[0].metric("NII at start (2024)", f"€{n0 / BN:.2f}bn")
    k[1].metric("Adverse NII 2027", f"€{res_a.nii['NII'].iloc[-1] / BN:.2f}bn", f"{res_a.nii['NII'].iloc[-1] / n0 - 1:+.1%}")
    k[2].metric("Largest adverse decline", f"{res_a.nii['NII'].min() / n0 - 1:+.1%}")
    k[3].metric("Baseline NII 2027", f"€{res_b.nii['NII'].iloc[-1] / BN:.2f}bn", f"{res_b.nii['NII'].iloc[-1] / n0 - 1:+.1%}")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=res_b.nii.index, y=res_b.nii["NII"] / BN, name="baseline", line_color=NAVY)
        fig.add_scatter(x=res_a.nii.index, y=res_a.nii["NII"] / BN, name="adverse", line_color=RED)
        fig.add_scatter(x=res_a.nii.index, y=res_a.nii["NII uncapped"] / BN, name="adverse before cap", line=dict(color=RED, dash="dot"))
        fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="€bn", legend=LEG, xaxis=dict(dtick=1), title="Net interest income")
        st.plotly_chart(fig, width="stretch")
    with c2:
        det = res_a.nii.drop(columns=["interest income", "interest expense", "NII", "NII uncapped"]) / MN
        d_ = det.iloc[-1] - det.iloc[0]
        fig = go.Figure(go.Bar(x=d_.values, y=d_.index, orientation="h", marker_color=[GREEN if v > 0 else RED for v in d_.values]))
        fig.update_layout(height=380, margin=dict(t=30, b=10), xaxis_title="€m", title="Adverse: change 2024 → 2027 by line")
        st.plotly_chart(fig, width="stretch")
    st.dataframe((pd.concat({"baseline": res_b.nii, "adverse": res_a.nii}) / MN).map(lambda v: f"{v:,.0f}"), width="stretch")
    st.caption("Drivers: the performing loan book shrinks as exposures default (defaulted loans earn only on their net carrying "
               "amount), the short rate falls back below its starting level by 2027 while deposit costs adjust only partially "
               "(low-β sight deposits), and wholesale funding reprices at the stressed spread.")

# ---------------------------------------------------------------- P&L
with tabs[6]:
    pl = pd.concat({"baseline": res_b.pnl.T, "adverse": res_a.pnl.T}, axis=1) / MN
    st.dataframe(pl.map(lambda v: f"{v:,.0f}"), width="stretch", height=h(pl))
    st.caption("€ million. Market shock (HFT positions, counterparty defaults, CVA) in 2025 only; FVOCI revaluation and the AVA "
               "increase go directly to CET1 (see Capital).")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for r, col in ((res_b, NAVY), (res_a, RED)):
            fig.add_bar(x=r.pnl.index, y=r.pnl["net profit"] / BN, name=r.scenario, marker_color=col)
        fig.update_layout(barmode="group", height=340, margin=dict(t=30, b=10), yaxis_title="€bn", legend=LEG, title="Net profit", xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for c_, col in (("operational losses", ORANGE), ("conduct losses", RED)):
            fig.add_bar(x=[f"{y} B" for y in res_b.pnl.index], y=-res_b.pnl[c_] / MN, name=c_, marker_color=col, legendgroup=c_)
            fig.add_bar(x=[f"{y} A" for y in res_a.pnl.index], y=-res_a.pnl[c_] / MN, name=c_, marker_color=col, legendgroup=c_, showlegend=False)
        fig.update_layout(barmode="stack", height=340, margin=dict(t=30, b=10), yaxis_title="€m", legend=LEG,
                          title="Operational and conduct losses (material conduct events booked in 2025 in the adverse)")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- capital
with tabs[7]:
    c1, c2 = st.columns(2)
    with c1:
        cap = res_a.capital
        fig = go.Figure()
        for c_, col in (("credit REA", NAVY), ("market REA", ORANGE), ("CCR & CVA REA", TEAL), ("operational REA", GREY)):
            fig.add_bar(x=cap.index, y=cap[c_] / BN, name=c_, marker_color=col)
        fig.add_scatter(x=cap.index, y=cap["REA SA-equivalent"] * 0.725 / BN, name="72.5 % × SA-equivalent", line=dict(color=RED, dash="dash"))
        cf = cap[cap.index > SC.START]
        fig.add_scatter(x=cf.index, y=cf["REA SA-equivalent"] * cf.index.map(lambda y: E.FLOOR_TRANSITIONAL.get(y, 0.725)) / BN,
                        name="transitional floor (50 / 55 / 60 %)", line=dict(color=RED, dash="dot"))
        fig.update_layout(barmode="stack", height=400, margin=dict(t=30, b=10), yaxis_title="€bn", legend=LEG, xaxis=dict(dtick=1),
                          title="Adverse REA and output floor")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        for r, col in ((res_b, NAVY), (res_a, RED)):
            fig.add_scatter(x=r.capital.index, y=r.capital["leverage ratio"], name=r.scenario, line=dict(color=col, width=3))
        fig.add_hline(y=0.03, line_dash="dash", line_color=GREY, annotation_text="3 % minimum")
        fig.update_layout(height=400, margin=dict(t=30, b=10), yaxis_tickformat=".1%", legend=LEG, xaxis=dict(dtick=1), title="Leverage ratio")
        st.plotly_chart(fig, width="stretch")
    ct = TP.capital(res_b, res_a)
    fm = {c: "{:,.0f}" for c in ct.columns if c not in ("year", "scenario")}
    fm.update({"CET1 ratio (transitional)": "{:.2%}", "CET1 ratio (fully loaded)": "{:.2%}", "leverage ratio": "{:.2%}"})
    st.dataframe(text(ct, fm), width="stretch", hide_index=True, height=h(ct))
    st.caption("€ million. CET1 movements: net profit − distributions + FVOCI revaluation − AVA increase − change in IRB shortfall. "
               "The output floor phases in at 50 % / 55 % / 60 % over 2025-2027 (CRR3 art. 465) and 72.5 % fully loaded.")

# ---------------------------------------------------------------- sensitivity & reverse
with tabs[8]:
    c1, c2 = st.columns(2)
    with c1:
        cont = run_contrib(bank, paths, shocks, fits, P, key)
        fig = go.Figure(go.Bar(x=cont.index, y=cont.values * 1e4, marker_color=[RED if v < 0 else GREEN for v in cont.values],
                               text=[f"{v * 1e4:+.0f}" for v in cont.values], textposition="outside"))
        fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="bp vs baseline (2027, fully loaded)",
                          title="One-at-a-time: adverse impact of each block alone")
        st.plotly_chart(fig, width="stretch")
        st.caption("Each block is stressed alone, the others staying on the baseline. The interaction term collects non-linear "
                   "effects: no tax relief and no distributions in loss years, REA changes, NII on defaulted assets.")
    with c2:
        tor = run_tornado(bank, paths, shocks, fits, P, key)
        fig = go.Figure()
        fig.add_bar(y=tor.index, x=tor["low"] * 1e4, orientation="h", name="parameter lower", marker_color=TEAL)
        fig.add_bar(y=tor.index, x=tor["high"] * 1e4, orientation="h", name="parameter higher", marker_color=ORANGE)
        fig.update_layout(barmode="overlay", height=380, margin=dict(t=30, b=10), xaxis_title="bp change of the 2027 adverse CET1 ratio",
                          legend=LEG, title="Methodology sensitivity (tornado)")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Reverse stress test")
    c1, c2 = st.columns([1, 2])
    with c1:
        thr = st.number_input("CET1 threshold (fully loaded, lowest point)", 0.03, 0.15, 0.080, 0.0025, format="%.4f",
                              help="For example the overall capital requirement (Pillar 1 + P2R + combined buffer) or the point of non-viability.")
        rv_ = run_reverse(bank, paths, shocks, fits, P, key, thr)
        if np.isfinite(rv_["scale"]):
            st.metric("Required severity (× adverse deviation from baseline)", f"{rv_['scale']:.2f}×")
            sc_path = SC.scaled(paths, rv_["scale"])
            st.metric("Implied cumulative GDP 2025-27", f"{np.prod(1 + sc_path['GDP growth'].iloc[1:] / 100) - 1:+.1%}")
            st.metric("Implied unemployment 2027", f"{sc_path['unemployment'].iloc[-1]:.1f} %")
            st.metric("Implied house prices, cumulative", f"{SC.cumulative(sc_path, 'HPI growth').iloc[-1] / 100 - 1:+.1%}")
        else:
            st.warning(rv_["note"])
        if rv_.get("note"):
            st.caption(rv_["note"])
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=rv_["grid"], y=rv_["values"], mode="lines+markers", line_color=NAVY, name="lowest CET1 ratio")
        fig.add_hline(y=thr, line_dash="dash", line_color=RED, annotation_text="threshold")
        fig.add_vline(x=1.0, line_dash="dot", line_color=GREY, annotation_text="EBA-style adverse")
        if np.isfinite(rv_["scale"]):
            fig.add_vline(x=rv_["scale"], line_color=RED)
        fig.update_layout(height=380, margin=dict(t=30, b=10), xaxis_title="scenario severity (0 = baseline, 1 = adverse)",
                          yaxis_tickformat=".1%", title="Lowest fully loaded CET1 ratio vs severity")
        st.plotly_chart(fig, width="stretch")
    st.caption("Severity scales the deviation of every macro variable from the baseline and the instantaneous market shocks; "
               "operational, fee and funding-spread stresses scale with it.")

# ---------------------------------------------------------------- templates
with tabs[9]:
    tables = TP.all_tables(res_b, res_a)
    which = st.selectbox("Template", list(tables), format_func=lambda k_: {
        "CSV_SUM": "CSV_SUM – summary of ratios", "CSV_CAP": "CSV_CAP – capital and REA", "CSV_P&L": "CSV_P&L – profit and loss",
        "CSV_CR_SCEN": "CSV_CR_SCEN – credit risk parameters", "CSV_CR_PROV": "CSV_CR_PROV – stages, flows and provisions",
        "CSV_CR_REA": "CSV_CR_REA – credit REA", "CSV_MR": "CSV_MR – market risk, CCR, CVA, AVA", "CSV_NII": "CSV_NII – net interest income",
        "CSV_OR": "CSV_OR – operational and conduct risk"}[k_])
    tb = tables[which]
    num = [c for c in tb.columns if pd.api.types.is_float_dtype(tb[c])]
    fmt = {c: ("{:.2%}" if ("ratio" in c or c in ("start", "end", "lowest", "default rate", "PD (12m, avg)", "LGD (avg)", "S3 coverage")) else "{:,.1f}") for c in num}
    st.dataframe(text(tb, fmt), width="stretch", hide_index=True, height=min(h(tb), 600))
    c1, c2 = st.columns(2)
    c1.download_button("Download all templates (Excel)", TP.to_excel(tables), "stress_test_templates.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button(f"Download {which} (CSV)", tb.to_csv(index=False).encode(), f"{which}.csv", "text/csv")
    st.caption("Simplified layouts inspired by the EBA result templates; amounts in € million. Not the official EBA templates.")

# ---------------------------------------------------------------- data & import
with tabs[10]:
    st.markdown("**Credit portfolio** — one row per portfolio × country (amounts in €, rates and parameters as decimals). "
                "Upload a file with the same columns to stress your own book.")
    st.download_button("Credit template (CSV)", D.credit_book().to_csv(index=False).encode(), "credit_template.csv", "text/csv")
    up = st.file_uploader("Credit CSV", type="csv")
    if up is not None and st.button("Load the credit file"):
        try:
            st.session_state["credit_import"] = D.validate_credit(pd.read_csv(up))
            st.rerun()
        except Exception as err:  # noqa: BLE001
            st.error(f"Import failed: {err}")
    if "credit_import" in st.session_state and st.button("Back to the synthetic bank"):
        del st.session_state["credit_import"]
        st.rerun()
    cr_ = bank.credit.copy()
    st.dataframe(text(cr_, {**{c: "{:,.0f}" for c in ["ead_s1", "ead_s2", "ead_s3"]}, "pd": "{:.2%}", "pd_reg": "{:.2%}", "lgd": "{:.0%}",
                             "s3_coverage": "{:.0%}", "rw_sa": "{:.0%}", "margin": "{:.2%}", "fixed_share": "{:.0%}"}), width="stretch", hide_index=True, height=360)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fair-value sensitivities**")
        st.dataframe(text(bank.market, {"sensitivity": "€{:,.0f}"}), width="stretch", hide_index=True, height=h(bank.market))
    with c2:
        st.markdown("**Funding and other interest-bearing items**")
        st.dataframe(text(bank.funding, {"amount": "€{:,.0f}", "beta": "{:.2f}", "spread": "{:.2%}", "fixed share": "{:.0%}", "repricing": "{:.0%}"}),
                     width="stretch", hide_index=True)
        st.markdown("**Starting point**")
        sp = pd.Series({"CET1": bank.cet1, "AT1": bank.at1, "T2": bank.t2, "market REA": bank.rea_market, "CCR & CVA REA": bank.rea_ccr_cva,
                        "operational REA": bank.rea_op, "leverage exposure": bank.lev_exposure, "fees": bank.fees, "expenses": bank.expenses,
                        "trading income": bank.trading_income, "op. losses (hist. avg)": bank.oprisk_hist, "conduct (hist. avg)": bank.conduct_hist,
                        "material conduct events": bank.material_conduct, "AVA": bank.ava}, name="€").to_frame()
        st.dataframe(sp.map(lambda v: f"{v / MN:,.0f} m"), width="stretch")

# ---------------------------------------------------------------- methodology
with tabs[11]:
    st.markdown(r"""
### Framework
* **Constrained bottom-up** projections over a three-year horizon under a common baseline and adverse scenario, with a
  **static balance sheet**: no growth, no change in business mix, maturing assets replaced like-for-like, no management actions.
  Defaulted assets are not replaced.
* Reference: methodology of the 2025 EU-wide stress test (64 banks, horizon 2025-2027, adverse scenario with a 6.3 % fall of EU
  GDP; aggregate CET1 depletion of 370 bp to 12 % at the end of 2027, credit losses €394bn). The draft methodology for 2027
  (published 11 June 2026) cuts the data points by 55 %, adds a separate climate module and keeps the centralised (top-down)
  projection of NII and net fee and commission income.

### Credit risk
* **Satellite models** map the scenario to point-in-time PDs (logit shift of the starting PD); LGD of secured portfolios
  moves with the collateral index, unsecured LGD with unemployment.
* **IFRS 9 stages**: annual transitions S1→S2, S1→S3, S2→S3, S2→S1 driven by the stressed PD; by default no cure from S3.
  Provisions: 12-month ECL (S1), lifetime ECL (S2), coverage × exposure (S3); impairment = change in the provision stock.
* **REA**: SA risk weights on net exposures (defaulted: 100 % / 150 %); IRB risk-weight function with a regulatory PD that
  migrates with the point-in-time PD (cyclicality parameter). Credit REA cannot fall below its starting value. IRB shortfall
  (expected loss above provisions) is deducted from CET1.

### Market risk, CCR and CVA
* Instantaneous shock at the start of the horizon: revaluation of held-for-trading (P&L) and FVOCI (OCI) positions, increase of
  prudent-valuation adjustments (CET1 deduction), default of the two most vulnerable of the ten largest counterparties, CVA on spread
  widening. Client trading revenues are lost in year 1 and reduced afterwards. Market and CCR REA held constant.

### Net interest income
* Loans: fixed and floating parts, client margin (not increased in the adverse), defaulted loans earning on their net amount.
* Liabilities: deposit pass-through (β) with a zero floor, wholesale funding repricing at the stressed funding spread.
* Adverse NII capped at its starting level.

### Operational and conduct risk, other P&L
* Non-conduct operational losses: historical average × adverse multiplier; conduct: material events (booked in year 1 in the
  adverse) plus non-material losses at a multiple of their history. Operational REA held constant.
* Fees and commissions capped at the starting point (baseline) and reduced in the adverse; administrative expenses indexed on
  scenario inflation and floored at their starting level.
* Common tax rate on profits; no tax credit on losses by default; distributions = payout × positive net profit.

### Capital and ratios
CET1_t = CET1_{t−1} + net profit − distributions + OCI − ΔAVA − ΔIRB shortfall. REA = credit + market + CCR/CVA + operational,
with the CRR3 output floor (transitional 50 / 55 / 60 %, fully loaded 72.5 %). Leverage ratio = Tier 1 / static exposure measure.
The depletion waterfall expresses each driver in percentage points of the starting REA; the residual is the REA effect.

### Not in scope
Securitisations, sovereign exposures in the banking book beyond the credit model, behavioural modelling of deposits, the 2027
climate module, consolidation and minority interests, P2R / P2G calibration.
""")
