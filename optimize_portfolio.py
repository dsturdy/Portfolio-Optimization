from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from pathlib import Path

from case_io import (
    load_current_portfolio_from_case,
    load_investable_universe_from_case,
)
from ticker_meta import TICKER_META

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CASE_FILE = BASE_DIR / "case_inputs.xlsx"
SPLICED_RET_FILE = DATA_DIR / "spliced_monthly_returns.csv"
SPLICED_PRICE_FILE = DATA_DIR / "spliced_price_panel_base_funds.csv"
OUT_WEIGHTS_FILE = DATA_DIR / "optimized_weights.csv"

"""
λ calibrated so each risk profile targets intuitive historical volatility ranges 
(Conservative < Moderate < Aggressive) using quarterly-rebalanced backtests.
"""

RISK_RETURN_TRADEOFF = {
    "Conservative": 8.0,
    "Moderate":     4.0,
    "Aggressive":   2.0,
}

TARGET_VOL_BY_PROFILE = {
    "Conservative": 0.07,
    "Moderate":     0.11,
    "Aggressive":   0.14,
}

EQUITY_CAP_BY_PROFILE = {
    "Conservative": 0.50,
    "Moderate":     0.70,
    "Aggressive":   0.90,
}

ILLIQUID_CAP_BY_PROFILE = {
    "Conservative": 0.10,
    "Moderate":     0.15,
    "Aggressive":   0.20,
}

CASH_CAP_BY_PROFILE = {
    "Conservative": 0.45,
    "Moderate":     0.30,
    "Aggressive":   0.15,
}

# ---------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------
def load_spliced_returns(ret_path: Path = SPLICED_RET_FILE) -> pd.DataFrame:
    """Monthly returns for optimizer / backtests."""
    rets = pd.read_csv(ret_path, index_col=0)
    rets.index = pd.to_datetime(rets.index)
    return rets


def load_spliced_prices(price_path: Path = SPLICED_PRICE_FILE) -> pd.DataFrame:
    """Daily spliced price panel for NAV paths."""
    prices = pd.read_csv(price_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    return prices


def estimate_moments(rets: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, float]:
    """
    Estimate annualized mean, covariance and RF from MONTHLY returns.
    """
    mu_m = rets.mean()
    cov_m = rets.cov()

    mu_a = 12.0 * mu_m
    cov_a = 12.0 * cov_m

    if "SGOV" in rets.columns:
        rf_monthly = rets["SGOV"].mean()
        rf_annual = float(12.0 * rf_monthly)
    else:
        rf_annual = 0.02

    return mu_a, cov_a, rf_annual


# ---------------------------------------------------------------------
"""
CORE OPTIMIZER (Part 1 of case study)

Optimizing for weights: w_i = weight in fund i (long-only, fully invested: w_i ≥ 0, Σ w_i = 1)

Constraints:
   - Asset-group min / max bands from case file (per sleeve)
   - Risk-profile caps:
       * Equity cap (Conservative / Moderate / Aggressive)
       * Cash cap (Conservative / Moderate / Aggressive)
       * Illiquid cap (Conservative / Moderate / Aggressive)

   - Optional turnover cap vs current weights 
   - Optional sleeve exclusions:
       * Real assets (FEREX)
       * Non-US equity
       * Short-duration credit (FDUIX)

Objectives:
   - "risk_targeted":
       Maximize μᵀw
       Subject to wᵀΣw ≤ σ_target²
       Maximize expected return while targeting a given portfolio volatility. 
   - "risk_return_tradeoff":
       Minimize λ·(wᵀΣw) − (μ − r_f·1)ᵀw
       Balance excess return and variance using a weighted risk penalty.
"""

def optimize_portfolio(
    risk_profile: str = "Moderate",
    objective: str = "risk_targeted",
    equity_cap: float | None = None,
    illiquid_cap: float | None = None,
    turnover_budget: float | None = 0.30,
    include_real_assets: bool = True,
    include_non_us_equity: bool = True,
    include_short_duration_credit: bool = True,
    window: str = "full",
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """Single ran optimization on the spliced monthly return panel."""

    # ---- Load inputs ----
    cp_current = load_current_portfolio_from_case(CASE_FILE)
    universe = load_investable_universe_from_case(CASE_FILE)
    rets = load_spliced_returns(SPLICED_RET_FILE)

    # ---- Window choice ----
    if window == "last5":
        cutoff = rets.index.max() - pd.DateOffset(years=5)
        rets = rets[rets.index >= cutoff]

    # ---- Align investable tickers with available returns ----
    common_tickers = sorted(set(universe["Ticker"]) & set(rets.columns))
    if not common_tickers:
        raise ValueError("No overlap between investable universe and return panel.")

    unknown = [t for t in common_tickers if t not in TICKER_META]
    if unknown:
        print("Warning: missing metadata for:", unknown)

    universe = universe.set_index("Ticker").reindex(common_tickers).reset_index()

    if "Fund Name" in universe.columns and "FundName" not in universe.columns:
        universe = universe.rename(columns={"Fund Name": "FundName"})

    rets = rets[common_tickers]

    # ---- Map group constraints (allocation bands) ----
    # Treat blank / None as "no explicit band"; convert to floats so None -> NaN.
    cp_bands = cp_current.copy()
    cp_bands["MinPct"] = cp_bands["MinPct"].astype("float")
    cp_bands["MaxPct"] = cp_bands["MaxPct"].astype("float")

    band_map = (
        cp_bands.groupby("AssetGroup")[["MinPct", "MaxPct"]]
        .agg({"MinPct": "min", "MaxPct": "max"})
    )

    universe = universe.merge(
        band_map,
        how="left",
        left_on="AssetGroup",
        right_index=True,
    )

    # If an asset group has no explicit band anywhere, treat it as unconstrained 0–100%.
    universe["MinPct"] = universe["MinPct"].fillna(0.0)
    universe["MaxPct"] = universe["MaxPct"].fillna(1.0)


    # ---- Current weights (turnover reference) ----
    current_w = (
        cp_current.set_index("Ticker")["Weight"]
        .reindex(common_tickers)
        .fillna(0.0)
        .values
    )

    # ---- Moments (MONTHLY → annualized) ----
    mu, cov, rf = estimate_moments(rets)
    mu_vec = mu[common_tickers].values
    cov_mat = cov.loc[common_tickers, common_tickers].values
    n = len(common_tickers)

    # -----------------------------------------------------------------
    # Decision variable + constraints
    # -----------------------------------------------------------------
    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1.0]

    # ---- Per-fund allocation bands from current-portfolio table ----
    fund_bands = (
        cp_current.set_index("Ticker")[["MinPct", "MaxPct"]]
        .reindex(common_tickers)
    )

    for i, t in enumerate(common_tickers):
        lo = fund_bands.at[t, "MinPct"]
        hi = fund_bands.at[t, "MaxPct"]

        # None / NaN means "no explicit bound for this fund"
        if pd.notna(lo):
            constraints.append(w[i] >= float(lo))
        if pd.notna(hi):
            constraints.append(w[i] <= float(hi))

    # ---- Group allocation bands (sleeve-level) ----
    for grp, sub in universe.groupby("AssetGroup"):
        idx = [common_tickers.index(t) for t in list(sub["Ticker"])]
        constraints += [
            cp.sum(w[idx]) >= float(sub["MinPct"].min()),
            cp.sum(w[idx]) <= float(sub["MaxPct"].max()),
        ]


    # Equity cap (from risk profile if not set)
    if equity_cap is None:
        equity_cap = EQUITY_CAP_BY_PROFILE.get(risk_profile, 0.70)

    equity_idx = [
        i for i, t in enumerate(common_tickers)
        if TICKER_META.get(t, {}).get("asset_class", "").startswith("equity")
    ]
    if equity_idx:
        constraints.append(cp.sum(w[equity_idx]) <= equity_cap)

    # Illiquid / quarterly liquidity cap
    if illiquid_cap is None:
        illiquid_cap = ILLIQUID_CAP_BY_PROFILE.get(risk_profile, 0.15)

    if illiquid_cap is not None:
        illiquid_idx = [
            i for i, t in enumerate(common_tickers)
            if TICKER_META.get(t, {}).get("liquidity", "daily") != "daily"
        ]
        if illiquid_idx:
            constraints.append(cp.sum(w[illiquid_idx]) <= illiquid_cap)


    # Cash cap (SGOV)
    cash_idx = [
        i for i, t in enumerate(common_tickers)
        if TICKER_META.get(t, {}).get("asset_class") == "cash"
    ]
    cash_cap = CASH_CAP_BY_PROFILE.get(risk_profile, 0.10)

    if cash_idx:
        constraints.append(cp.sum(w[cash_idx]) <= cash_cap)

    # Sleeve toggles
    if not include_real_assets:
        for t in common_tickers:
            if TICKER_META.get(t, {}).get("asset_class") == "real_assets":
                constraints.append(w[common_tickers.index(t)] == 0.0)

    if not include_non_us_equity:
        for t in common_tickers:
            if TICKER_META.get(t, {}).get("asset_class") == "equity_non_us":
                constraints.append(w[common_tickers.index(t)] == 0.0)

    if not include_short_duration_credit:
        for t in common_tickers:
            if TICKER_META.get(t, {}).get("asset_class") == "fixed_short_credit":
                constraints.append(w[common_tickers.index(t)] == 0.0)

    # Turnover constraint vs current portfolio
    if turnover_budget is not None and turnover_budget > 0:
        # L1 distance counts both sells and buys, so ||w - w0||_1 ≈ 2 × turnover
        l1_cap = 2.0 * turnover_budget
        constraints.append(cp.norm1(w - current_w) <= l1_cap)

    # -----------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------
    excess_mu = mu_vec - rf

    lambda_val: float | None = None
    target_vol: float | None = None

    if objective == "risk_targeted":
        target_vol = TARGET_VOL_BY_PROFILE[risk_profile]

        # convex risk constraint: variance <= target_vol^2
        constraints.append(cp.quad_form(w, cov_mat) <= target_vol ** 2)

        # maximize expected return given that risk cap
        obj = cp.Minimize(-(w @ mu_vec))

    elif objective == "risk_return_tradeoff":
        lambda_val = RISK_RETURN_TRADEOFF[risk_profile]
        obj = cp.Minimize(
            lambda_val * cp.quad_form(w, cov_mat) - (w @ excess_mu)
        )

    else:
        raise ValueError(f"Unknown objective: {objective}")

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------
    print("\n--- Constraint Summary ----------------------------------")
    print(f"Risk Profile        : {risk_profile}")
    print(f"Equity Cap          : {equity_cap:.0%}")
    print(f"Illiquid Cap        : {illiquid_cap if illiquid_cap is not None else 'None'}")
    print(f"Turnover Budget     : {turnover_budget if turnover_budget is not None else 'None'}")
    print(f"Include Real Assets : {include_real_assets}")
    print(f"Include Non-US Eq   : {include_non_us_equity}")
    print(f"Include Short Crdt  : {include_short_duration_credit}")
    print(f"Window              : {window}")
    print("----------------------------------------------------------\n")

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)

    if prob.status in ("infeasible", "infeasible_inaccurate"):
        raise RuntimeError(f"Optimization infeasible: {prob.status}")

    weights = pd.Series(w.value, index=common_tickers).clip(lower=0)
    weights /= weights.sum()

    port_ret = float(weights @ mu_vec)
    port_vol = float(np.sqrt(weights @ cov_mat @ weights))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan

    summary = {
        "Status": prob.status,
        "ExpectedReturn": port_ret,
        "Volatility": port_vol,
        "Sharpe": sharpe,
        "RiskProfile": risk_profile,
        "Objective": objective,
        "EquityCap": equity_cap,
        "IlliquidCap": illiquid_cap,
        "CashCap": cash_cap,
        "TurnoverBudget": turnover_budget,
        "Window": window,
        "RiskFreeAnnual": rf,
        "Lambda": lambda_val,  # None for risk_targeted
        "TargetVol": target_vol,  # None for risk_return_tradeoff
    }

    comp = pd.DataFrame(
        {
            "CurrentWeight": pd.Series(current_w, index=common_tickers),
            "OptimizedWeight": weights,
        }
    )
    comp["Change"] = comp["OptimizedWeight"] - comp["CurrentWeight"]

    extra_cols = ["AssetGroup"]
    if "FundName" in universe.columns:
        extra_cols.append("FundName")

    comp = comp.join(universe.set_index("Ticker")[extra_cols])

    return weights, summary, comp


# ---------------------------------------------------------------------
# REUSE OPTIMIZER FOR HISTORICAL BACKTESTING
# ---------------------------------------------------------------------
def run_optimizer_on_returns(
    rets: pd.DataFrame,
    risk_profile: str = "Moderate",
    objective: str | None = None,
    equity_cap: float | None = None,
    illiquid_cap: float | None = None,
    turnover_budget: float | None = 0.30,
    include_real_assets: bool = True,
    include_non_us_equity: bool = True,
    include_short_duration_credit: bool = True,
) -> tuple[pd.Series, dict]:
    """
    Re-run the optimizer on an arbitrary MONTHLY return panel.
    Powers rolling backtests without duplicating constraint logic.
    """
    global load_spliced_returns
    orig_loader = load_spliced_returns

    def _tmp_loader(_=None):
        return rets

    load_spliced_returns = _tmp_loader

    try:
        w, summary, _ = optimize_portfolio(
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=equity_cap,
            illiquid_cap=illiquid_cap,
            turnover_budget=turnover_budget,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_duration_credit,
            window="full",
        )
    finally:
        load_spliced_returns = orig_loader

    return w, summary

# ---------------------------------------------------------------------
# REBALANCE INDEX HELPER (MONTHLY DATE INDEX)
# ---------------------------------------------------------------------
def _rebalance_indices(
    dates: pd.DatetimeIndex,
    start_months: int,
    freq_months: int,
) -> list[int]:

    n_months = len(dates)
    if n_months < start_months:
        raise ValueError(
            f"Not enough history ({n_months} months) for start_months={start_months}"
        )

    start_ix = start_months - 1  # 0-based
    rebalance_ixs = list(range(start_ix, n_months, freq_months))
    if not rebalance_ixs:
        raise ValueError("No rebalance points computed.")
    return rebalance_ixs


# ---------------------------------------------------------------------
# QUARTERLY BACKTEST – DAILY NAV USING DAILY PRICES
# ---------------------------------------------------------------------
def backtest_quarterly_daily(
    rets: pd.DataFrame,
    mode: str = "static",          # "static" or "rolling"
    lookback_months: int = 60,
    start_months: int | None = None,
    rebalance_freq: str = "quarterly",
    risk_profile: str = "Moderate",
    objective: str = "min_var",
    equity_cap: float | None = None,
    illiquid_cap: float | None = 0.15,
    turnover_budget: float | None = None,
    include_real_assets: bool = True,
    include_non_us_equity: bool = True,
    include_short_duration_credit: bool = True,
    override_weights: pd.Series | None = None,
) -> dict:
    """
    Quarterly rebalancing with DAILY NAV

    - For each rebalance month:
        * run optimizer on trailing `lookback_months` of MONTHLY returns
        * set holdings to those target weights
    - Between rebalances:
        * compound holdings using DAILY returns from the spliced price panel.
    """
    if mode not in {"static", "rolling"}:
        raise ValueError("mode must be 'static' or 'rolling'.")

    # Monthly layer for schedule + optim windows
    rets = rets.sort_index()
    monthly_dates = rets.index
    tickers = list(rets.columns)

    # ==========================================================
    # STATIC + override_weights  ->  ignore lookback completely
    # ==========================================================
    if mode == "static" and override_weights is not None:

        # Align weights to tickers
        w = pd.Series(override_weights, index=tickers).reindex(tickers).fillna(0.0)

        # Normalize (safety)
        if w.sum() != 0:
            w = w / w.sum()

        # Convert 1-based month index (app.py) to 0-based python index
        if start_months is None or start_months < 1:
            start_idx = 0
        else:
            start_idx = start_months - 1

        # Slice returns from that month forward
        rets_bt = rets.iloc[start_idx:].copy()

        # Load daily prices (same window)
        daily_prices = load_spliced_prices()
        daily_prices = daily_prices[tickers]
        daily_prices = daily_prices[
            (daily_prices.index >= rets_bt.index[0]) &
            (daily_prices.index <= rets_bt.index[-1])
            ]
        daily_prices = daily_prices.dropna(how="all")
        daily_rets = daily_prices.pct_change()

        # Fixed-weight portfolio — rebalancing to w implicitly each day
        port_rets = (daily_rets @ w).fillna(0.0)

        nav = (1.0 + port_rets).cumprod()
        nav = nav / nav.iloc[0] * 10000
        nav.name = "nav"

        return {"nav": nav}

    if start_months is None:
        start_months = lookback_months

    # Map string frequency -> number of months between rebalances
    freq_map = {
        "quarterly": 3,
        "semiannual": 6,
        "annual": 12,
    }
    if rebalance_freq not in freq_map:
        raise ValueError(
            f"rebalance_freq must be one of {list(freq_map.keys())}, "
            f"got {rebalance_freq!r}"
        )

    freq_months = freq_map[rebalance_freq]
    rebalance_month_ixs = _rebalance_indices(
        monthly_dates,
        start_months=start_months,
        freq_months=freq_months,
    )
    first_month_ix = rebalance_month_ixs[0]


    # Daily prices / returns for NAV compounding
    daily_prices = load_spliced_prices()
    # Restrict to same tickers + overlapping date range
    daily_prices = daily_prices[tickers]
    daily_prices = daily_prices[
        (daily_prices.index >= monthly_dates[0]) &
        (daily_prices.index <= monthly_dates[-1])
    ]
    daily_prices = daily_prices.dropna(how="all")
    daily_rets = daily_prices.pct_change()

    dates_daily = daily_prices.index
    n_days = len(dates_daily)

    # Map each rebalance *month* to a *daily* index (last daily <= month-end)
    rebalance_daily_ixs: list[int] = []
    for m_ix in rebalance_month_ixs:
        m_date = monthly_dates[m_ix]
        mask = dates_daily <= m_date
        if not mask.any():
            continue
        d_ix = np.where(mask)[0][-1]
        rebalance_daily_ixs.append(d_ix)

    if not rebalance_daily_ixs:
        raise ValueError("No valid daily rebalance dates found.")

    # Helper: optimizer at a rebalance month (using trailing monthly window)
    def _run_optimize_for_rebalance(m_ix: int) -> pd.Series:
        """
        If override_weights is provided, just use those (current-portfolio backtest).
        Otherwise, run the normal optimizer.

        - For mode="static": use the FULL user-selected history (rets) to set one
          static set of weights.
        - For mode="rolling": use a trailing `lookback_months` window ending
          just before the rebalance month.
        """
        if override_weights is not None:
            return override_weights.reindex(tickers).fillna(0.0)

        if mode == "static":
            # Static strategy uses the full selected window (full vs last5),
            # independent of the rolling lookback.
            window_rets = rets
        else:
            # Rolling strategy: trailing window before this rebalance month
            end_ix = m_ix - 1
            if end_ix < 0:
                raise ValueError(
                    "Need at least one month of history before first rebalance."
                )

            window_len = min(lookback_months, end_ix + 1)
            start_ix = end_ix + 1 - window_len
            window_rets = rets.iloc[start_ix : end_ix + 1]

        w_opt, _summary = run_optimizer_on_returns(
            window_rets,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=equity_cap,
            illiquid_cap=illiquid_cap,
            turnover_budget=turnover_budget,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_duration_credit,
        )
        return w_opt.reindex(tickers).fillna(0.0)

    # Ensure month + day rebalance lists have same length
    if len(rebalance_daily_ixs) != len(rebalance_month_ixs):
        L = min(len(rebalance_daily_ixs), len(rebalance_month_ixs))
        rebalance_daily_ixs = rebalance_daily_ixs[:L]
        rebalance_month_ixs = rebalance_month_ixs[:L]

    # Initial state
    current_target_w = None
    if mode == "static":
        current_target_w = _run_optimize_for_rebalance(first_month_ix)

    nav_daily = pd.Series(index=dates_daily, dtype=float)
    weights_daily = pd.DataFrame(index=dates_daily, columns=tickers, dtype=float)
    target_w_history = pd.DataFrame(index=dates_daily, columns=tickers, dtype=float)

    current_nav = 10000
    current_holdings = None
    current_weights = None

    reb_ptr = 0
    next_reb_day_ix = rebalance_daily_ixs[reb_ptr]
    next_reb_month_ix = rebalance_month_ixs[reb_ptr]
    first_reb_day_ix = next_reb_day_ix

    # Daily simulation
    for d_ix in range(first_reb_day_ix, n_days):
        # Rebalance at start of day if needed
        if d_ix == next_reb_day_ix:
            if mode == "rolling" or current_target_w is None:
                current_target_w = _run_optimize_for_rebalance(next_reb_month_ix)

            if current_holdings is None:
                current_holdings = current_nav * current_target_w.values
            else:
                current_holdings = current_nav * current_target_w.values

            current_weights = current_target_w.values
            target_w_history.iloc[d_ix] = current_target_w

            if reb_ptr < len(rebalance_daily_ixs) - 1:
                reb_ptr += 1
                next_reb_day_ix = rebalance_daily_ixs[reb_ptr]
                next_reb_month_ix = rebalance_month_ixs[reb_ptr]
            else:
                next_reb_day_ix = n_days + 1  # no more rebalances

        # Apply daily returns
        day_ret_vec = daily_rets.iloc[d_ix].values
        if current_holdings is None:
            nav_daily.iloc[d_ix] = current_nav
            continue

        current_holdings = current_holdings * (1.0 + day_ret_vec)
        current_nav = float(current_holdings.sum())
        current_weights = current_holdings / current_nav

        nav_daily.iloc[d_ix] = current_nav
        weights_daily.iloc[d_ix] = current_weights

    nav_daily = nav_daily.dropna()
    weights_daily = weights_daily.dropna(how="all")
    target_w_history = target_w_history.dropna(how="all")

    return {
        "nav": nav_daily,
        "weights": weights_daily,
        "target_weights": target_w_history,
        "rebalance_dates": nav_daily.index.intersection(target_w_history.index),
        "mode": mode,
        "lookback_months": lookback_months,
        "start_months": start_months,
        "rebalance_freq": rebalance_freq,
        "risk_profile": risk_profile,
        "objective": objective,
    }

def make_policy_60_40(rets: pd.DataFrame) -> pd.Series:
    """
    Build a policy 60/40 benchmark:
    60% URTH (MSCI World proxy)
    40% AGG  (US Agg proxy)
    """
    w = pd.Series(0.0, index=rets.columns, dtype=float)

    if "URTH" in rets.columns:
        w["URTH"] = 0.60
    if "AGG" in rets.columns:
        w["AGG"] = 0.40

    # If only one exists, put 100% there (failsafe)
    if w.sum() > 0:
        w = w / w.sum()

    return w

# ---------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------
def main():
    print("Running optimization using spliced returns...")
    weights, summary, comp = optimize_portfolio()

    comp.index.name = "Ticker"
    comp.to_csv(OUT_WEIGHTS_FILE)

    print(f"\nSaved optimized weights -> {OUT_WEIGHTS_FILE}\n")

    print("Portfolio statistics (annualized):")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>15}: {v: .4f}")
        else:
            print(f"  {k:>15}: {v}")

    print("\nTop allocation changes:")
    pretty = (100 * comp[["CurrentWeight", "OptimizedWeight", "Change"]]).round(2)
    print(pretty.sort_values("Change", ascending=False))


if __name__ == "__main__":
    main()



