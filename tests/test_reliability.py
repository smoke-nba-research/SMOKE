"""Shrinkage estimator checks."""

from __future__ import annotations

import numpy as np

from src.validate.reliability import dersimonian_laird, empirical_bayes_shrink
import pandas as pd


def test_dersimonian_laird_matches_hand_computation():
    x = np.array([0.02, -0.01, 0.05, 0.00])
    se = np.array([0.01, 0.02, 0.01, 0.015])
    w = 1 / se**2
    fixed = (w * x).sum() / w.sum()
    q = (w * (x - fixed) ** 2).sum()
    c = w.sum() - (w**2).sum() / w.sum()
    tau2_expected = max(0.0, (q - 3) / c)
    mu, tau2 = dersimonian_laird(x, se)
    assert abs(tau2 - tau2_expected) < 1e-12
    w_star = 1 / (se**2 + tau2)
    assert abs(mu - (w_star * x).sum() / w_star.sum()) < 1e-12


def test_no_heterogeneity_gives_zero_tau_and_full_shrinkage():
    x = np.array([0.001, -0.001, 0.0005, -0.0005])
    se = np.array([0.05, 0.05, 0.05, 0.05])
    _, tau2 = dersimonian_laird(x, se)
    assert tau2 == 0.0
    players = pd.DataFrame({"smoke_rate": x, "smoke_rate_se": se, "shots": [200] * 4})
    out = empirical_bayes_shrink(players)
    assert (out["reliability"] == 0).all()
    assert np.allclose(out["smoke_rate_shrunk"], out.attrs["mu"])


def test_precise_players_shrink_less():
    # real spread between players, so tau-squared is positive; the first two players
    # report the same rate with very different precision
    players = pd.DataFrame(
        {
            "smoke_rate": [0.05, 0.05, -0.04, 0.01, -0.02],
            "smoke_rate_se": [0.005, 0.03, 0.01, 0.01, 0.01],
            "shots": [900, 100, 500, 500, 500],
        }
    )
    out = empirical_bayes_shrink(players)
    assert out.attrs["tau"] > 0
    assert out.loc[0, "reliability"] > out.loc[1, "reliability"]
    assert abs(out.loc[0, "smoke_rate_shrunk"] - 0.05) < abs(out.loc[1, "smoke_rate_shrunk"] - 0.05)
