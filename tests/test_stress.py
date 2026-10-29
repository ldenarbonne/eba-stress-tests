import numpy as np
import pandas as pd
import pytest

from stress import credit as CR
from stress import data as D
from stress import engine as E
from stress import satellite as SAT
from stress import scenario as SC
from stress import templates as TP


@pytest.fixture(scope="module")
def setup():
    bank = D.generate()
    paths = SC.default_paths()
    shocks = SC.default_market_shocks()
    fits = SAT.fit_all(SAT.history(9))
    P = E.Params()
    return bank, paths, shocks, fits, P


@pytest.fixture(scope="module")
def res(setup):
    bank, paths, shocks, fits, P = setup
    return E.run(bank, paths, shocks, fits, P, "baseline"), E.run(bank, paths, shocks, fits, P, "adverse")


def test_scenario_calibration():
    p = SC.default_paths()
    gdp = np.prod(1 + p["adverse"]["GDP growth"].iloc[1:] / 100) - 1
    assert gdp == pytest.approx(-0.063, abs=0.003)
    assert p["adverse"]["unemployment"].iloc[-1] - p["baseline"]["unemployment"].iloc[-1] == pytest.approx(6.0, abs=0.3)


def test_satellite_recovers_signs():
    fits = SAT.fit_all(SAT.history(9))
    for p, f in fits.items():
        assert f.sign_ok().all(), p
        assert f.r2 > 0.7


def test_static_balance_sheet(res):
    for r in res:
        ead = r.credit.groupby("year")["EAD"].sum()
        assert np.allclose(ead, ead.iloc[0])


def test_adverse_worse_than_baseline(res):
    b, a = res
    assert a.pnl["impairments"].sum() < b.pnl["impairments"].sum()
    assert a.capital["CET1 ratio (fully loaded)"].iloc[-1] < b.capital["CET1 ratio (fully loaded)"].iloc[-1]
    assert a.nii["NII"].iloc[-1] < b.nii["NII"].iloc[-1]
    assert (a.nii["NII"].iloc[1:] <= a.nii["NII"].iloc[0] + 1e-6).all()       # cap


def test_waterfall_adds_up(res):
    w = res[1].waterfall
    assert w.sum() == pytest.approx(w.attrs["end"] - w.attrs["start"], abs=1e-12)


def test_capital_identity(res):
    a = res[1]
    c, p = a.capital, a.pnl
    d = c["CET1"].diff().iloc[1:].to_numpy()
    expect = (p["net profit"] + p["distributions"]).to_numpy() + c[["OCI (FVOCI)", "AVA", "IRB shortfall"]].iloc[1:].sum(axis=1).to_numpy()
    assert np.allclose(d, expect)


def test_credit_rea_floor_and_output_floor(res):
    for r in res:
        c = r.capital
        assert (c["credit REA"].iloc[1:] >= c["credit REA"].iloc[0] - 1).all()
        assert (c["REA fully loaded"] >= 0.725 * c["REA SA-equivalent"] - 1).all()
        assert (c["REA fully loaded"] >= c["REA transitional"] - 1).all()


def test_irb_rw_reference():
    rw = CR.irb_rw(np.array([0.01]), 0.45, 2.5, "corporate")
    assert rw[0] == pytest.approx(0.9244, abs=0.005)


def test_no_s3_cure_means_monotone_s3(res):
    s3 = res[1].credit.groupby("year")["S3"].sum()
    assert s3.is_monotonic_increasing


def test_ccr_two_most_vulnerable(setup):
    from stress import market as MK
    bank = setup[0]
    d = MK.ccr_defaults(bank.cpty, 2)
    assert len(d) == 2 and not d["type"].isin(["Sovereign", "CCP"]).any()
    assert set(d["counterparty"]) == {"Hedge fund D", "Corporate G"}


def test_reverse_stress_monotone(setup):
    bank, paths, shocks, fits, P = setup
    lows = [E.run(bank, paths, shocks, fits, P, "adverse", s).capital["CET1 ratio (fully loaded)"].min() for s in (0.5, 1.0, 2.0)]
    assert lows[0] > lows[1] > lows[2]
    rv = E.reverse_stress(bank, paths, shocks, fits, P, 0.08)
    assert 0 < rv["scale"] < 4
    assert E.run(bank, paths, shocks, fits, P, "adverse", rv["scale"]).capital["CET1 ratio (fully loaded)"].min() == pytest.approx(0.08, abs=5e-4)


def test_templates(res):
    t = TP.all_tables(*res)
    assert {"CSV_SUM", "CSV_CAP", "CSV_P&L", "CSV_MR"} <= set(t)
    assert len(TP.to_excel(t)) > 1000


def test_credit_validation():
    df = D.credit_book()
    assert len(D.validate_credit(df)) == len(df)
    with pytest.raises(ValueError):
        D.validate_credit(df.drop(columns="lgd"))
