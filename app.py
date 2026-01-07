# app.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from optimize_portfolio import optimize_portfolio, backtest_quarterly_daily, make_policy_60_40
from ticker_meta import TICKER_META
from analytics import (
    get_current_portfolio,
    get_returns_for_window,
    compute_group_risk_contrib,
    compute_max_drawdown,
)

# Reduce default Streamlit top padding so the app doesn't start halfway down the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Just creating a color palette to copy First Eagle's brand colors (definitely not perfect)
FE_COLORS = [
    "#1E293B",   # deep slate
    "#B9975B",   # gold
    "#6B7280",   # softer gray-blue
    "#A8B0BB",   # muted light grey-blue
    "#D6D3CE",   # warm light neutral
]

DEFAULT_LOOKBACK_MONTHS = 60          # used for rolling/backtests
DEFAULT_REBALANCE_FREQ = "quarterly"  # "quarterly" | "semiannual" | "annual"

FREQ_LABELS = {
    "quarterly": "Quarterly",
    "semiannual": "Semiannual",
    "annual": "Annual",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_css() -> None:
    # Load custom appearance from assets/styles.css
    # Keeps visual styling out of app.py.

    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    if css_path.exists():
        with css_path.open() as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def styled_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    # Making custom tables rather than defaulting to plotly's

    def band_columns(data: pd.DataFrame) -> pd.DataFrame:
        n_rows, n_cols = data.shape
        styles = np.full((n_rows, n_cols), "background-color: #FFFFFF", dtype=object)
        for j in range(n_cols):
            if j % 2 == 1:      # 2nd, 4th, 6th...
                styles[:, j] = "background-color: #F7FAFF"
        return pd.DataFrame(styles, index=data.index, columns=data.columns)

    styler = (
        df.style
        .hide(axis="index")                 # FULLY hides the index
        .apply(band_columns, axis=None)     # Column shading
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#E8F0FB"),
                        ("color", "#1E3A8A"),
                        ("font-weight", "700"),
                        ("border-bottom", "2px solid #BFD3F2"),
                        ("white-space", "nowrap"),
                    ],
                },
                {
                    # Auto width + nowrap for body cells
                    "selector": "td",
                    "props": [
                        ("white-space", "nowrap"),
                        ("max-width", "999px"),
                    ],
                },
            ]
        )
        .set_properties(
            **{
                "text-align": "left",
                "white-space": "nowrap",
            }
        )
        .format(precision=1)
    )

    return styler


@st.cache_data
def cached_current_portfolio(window: str = "full"):
    # Caches the stats of the excel portfolio
    return get_current_portfolio(window=window)


@st.cache_data
def cached_returns(window: str = "full"):
    # Caches the historical returns dataset
    return get_returns_for_window(window=window)


def make_group_bar_figure(
    comp: pd.DataFrame,
    current_col: str,
    optimized_col: str | None = None,
):

    # Bar chart by asset group for current and optimized weights.
    df = comp.copy()
    if "AssetGroup" not in df.columns:
        return px.bar()

    agg_cols = [current_col]
    if optimized_col is not None:
        agg_cols.append(optimized_col)

    by_grp = df.groupby("AssetGroup")[agg_cols].sum()
    by_grp = (100 * by_grp).round(2).reset_index()
    by_grp = by_grp.rename(columns={"AssetGroup": "Asset group"})

    if optimized_col is None:
        fig = px.bar(
            by_grp,
            x="Asset group",
            y=current_col,
            labels={current_col: "Weight"},
        )

        fig.update_traces(
            hovertemplate=(
                "Asset group = %{x}<br>"
                "Weight = %{y:.1f}%"
                "<extra></extra>"
            )
        )
    else:
        long_df = by_grp.melt(
            id_vars="Asset group",
            value_vars=[current_col, optimized_col],
            var_name="Portfolio",
            value_name="Weight (%)",
        )
        long_df["Portfolio"] = long_df["Portfolio"].map(
            {current_col: "Current", optimized_col: "Optimized"}
        )
        fig = px.bar(
            long_df,
            x="Asset group",
            y="Weight (%)",
            color="Portfolio",
            barmode="group",
            color_discrete_sequence=["#B9975B", "#1E293B"],  # current vs optimized
        )

        fig.update_traces(
            hovertemplate=(
                "Asset group = %{x}<br>"
                "Weight = %{y:.1f}%"
                "<extra></extra>"
            )
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        xaxis_title="",
    )
    return fig


def make_risk_contrib_pie(rc_df: pd.DataFrame):

    # Pie chart of risk contribution by asset group

    df = rc_df.copy().reset_index()
    df = df.rename(columns={"index": "Asset group"})

    df["Risk contribution (%)"] = df["Risk contribution (%)"].astype(float).round(1)

    # Filter out tiny slices
    threshold = 0.1
    df = df[df["Risk contribution (%)"].abs() > threshold]

    fig = px.pie(
        df,
        names="Asset group",
        values="Risk contribution (%)",
        hole=0.40,  # donut style
        color_discrete_sequence=FE_COLORS,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate=(
            "%{label}<br>"
            "Risk contribution = %{value:.1f}%"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=550,
        legend_title="Asset group",
    )

    return fig


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="First Eagle – Portfolio Optimization Program",
        layout="wide",
    )

    load_css()

    st.title("First Eagle – Portfolio Optimization Program")

    st.markdown(
        """
        Applies a **rules-based mean–variance optimizer** to the
        case study portfolio.

        - The **top section** shows the current portfolio as-is (based on the case file).
        - The **optimization section** updates once you adjust inputs in the sidebar
          and click **Run optimization**.
        """,
    )

    # ---------------------------------------------------------------
    # Sidebar controls
    # ---------------------------------------------------------------
    st.sidebar.header("Optimization controls")

    risk_profile = st.sidebar.selectbox(
        "Risk profile",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
    )

    objective = st.sidebar.radio(
        "Objective",
        ["risk_targeted", "risk_return_tradeoff"],
        index=0,
        format_func=lambda s: (
            "Risk-targeted return maximization"
            if s == "risk_targeted"
            else "Risk–return tradeoff"
        ),
    )

    window = st.sidebar.radio(
        "Return window",
        ["full", "last5"],
        index=0,
        format_func=lambda s: "Full history" if s == "full" else "Last 5 years",
    )

    # Returns for the chosen window (used for stats/backtests)
    rets_win = cached_returns(window=window)
    max_months = len(rets_win)

    # Fixed lookback (capped by available history) – tweak DEFAULT_LOOKBACK_MONTHS above
    lookback_months = min(DEFAULT_LOOKBACK_MONTHS, max_months)


    turnover_budget = st.sidebar.slider(
        "Reallocation limit",
        0.0,
        1.00,
        0.15,
        step=0.01,
    )

    with st.sidebar.expander("Advanced constraints"):

        include_real_assets = st.checkbox(
            "Include real assets (FEREX)",
            True,
        )

        include_non_us_equity = st.checkbox(
            "Include non-US equity",
            True,
        )

        include_short_credit = st.checkbox(
            "Include short-duration credit (FDUIX)",
            True,
        )

    # Backtest frequency (fixed; tweak DEFAULT_REBALANCE_FREQ above)
    rebalance_freq = DEFAULT_REBALANCE_FREQ
    freq_label = FREQ_LABELS[rebalance_freq]


    show_constraints = st.sidebar.checkbox(
        "Show constraint details",
        value=False,
    )

    run_clicked = st.sidebar.button("Run optimization")


    # ---------------------------------------------------------------
    # Current portfolio section (always shown)
    # ---------------------------------------------------------------
    base_summary, base_comp = cached_current_portfolio(window=window)

    # Current max drawdown
    cur_mdd = compute_max_drawdown(
        rets_win,
        base_comp["CurrentWeight"],
        list(base_comp.index),
    )

    st.subheader("Current portfolio")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected return", f"{base_summary['ExpectedReturn']:.2%}")
    c2.metric("Volatility", f"{base_summary['Volatility']:.2%}")
    c3.metric("Sharpe ratio", f"{base_summary['Sharpe']:.2f}")
    c4.metric("Max drawdown", f"{cur_mdd:.2%}" if np.isfinite(cur_mdd) else "N/A")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Current weights by fund**")

        cur_df = base_comp.copy()
        cur_df["Current (%)"] = (100 * cur_df["CurrentWeight"]).round(1)

        # Filter out microscopic weights so they don't take up space in the UI
        threshold = 0.01
        cur_df = cur_df[cur_df["Current (%)"].abs() > threshold]

        cur_df = cur_df.reset_index().rename(columns={"index": "Ticker"})
        cur_df = cur_df[["Fund", "Ticker", "AssetGroup", "Current (%)"]]
        cur_df = cur_df.rename(columns={"AssetGroup": "Asset group"})
        cur_df = cur_df.sort_values("Current (%)", ascending=False)

        st.table(styled_table(cur_df))

    with col_right:
        st.markdown("**Current allocation by asset group**")
        fig_current_bar = make_group_bar_figure(base_comp, current_col="CurrentWeight")
        st.plotly_chart(fig_current_bar, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------
    # Optimization section (after button click)
    # ---------------------------------------------------------------
    if run_clicked:
        try:
            with st.spinner("Solving optimization problem..."):
                weights, summary, comp = optimize_portfolio(
                    risk_profile=risk_profile,
                    objective=objective,
                    illiquid_cap=None,
                    turnover_budget=turnover_budget if turnover_budget > 0 else None,
                    include_real_assets=include_real_assets,
                    include_non_us_equity=include_non_us_equity,
                    include_short_duration_credit=include_short_credit,
                    window=window,
                )
        except RuntimeError as e:
            st.error(
                "The optimization was infeasible with your current settings.\n\n"
                "Try relaxing one or more of:\n"
                "- Reallocation limit (L1 distance vs current)\n"
                "- Illiquid sleeve cap\n"
                "- Asset-group min/max bands in the case file."
            )
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error while solving the optimization: {e}")
            st.stop()


        st.subheader("Optimized portfolio")

        # Max drawdown for optimized portfolio
        opt_mdd = compute_max_drawdown(
            rets_win,
            weights,
            list(comp.index),
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected return", f"{summary['ExpectedReturn']:.2%}")
        m2.metric("Volatility", f"{summary['Volatility']:.2%}")
        m3.metric("Sharpe ratio", f"{summary['Sharpe']:.2f}")
        m4.metric(
            "Max drawdown",
            f"{opt_mdd:.2%}" if np.isfinite(opt_mdd) else "N/A",
        )

        # ---------------- Constraint summary (optional debug) --------
        if show_constraints:
            with st.expander("Constraint summary (details)", expanded=False):

                cons_rows = [
                    ("Risk profile", summary["RiskProfile"]),
                    ("Objective", summary["Objective"]),
                    ("Equity cap", f"{summary['EquityCap']:.0%}"),
                    (
                        "Illiquid cap",
                        f"{summary['IlliquidCap']:.0%}"
                        if summary["IlliquidCap"] is not None
                        else "None",
                    ),
                    (
                        "Cash / T-bills cap",
                        f"{summary['CashCap']:.0%}",
                    ),
                    (
                        "Reallocation limit (L1)",
                        f"{summary['TurnoverBudget']:.2f}"
                        if summary["TurnoverBudget"] is not None
                        else "None",
                    ),
                    (
                        "Risk-free rate (annualized)",
                        f"{summary['RiskFreeAnnual']:.2%}",
                    ),
                ]

                # 👇 show λ ONLY when we're using the λ-weighted objective
                if summary.get("Lambda") is not None:
                    cons_rows.append(
                        ("Risk aversion (λ)", f"{summary['Lambda']:.2f}")
                    )

                # 👇 show target volatility ONLY for risk-targeted objective
                if summary.get("TargetVol") is not None:
                    cons_rows.append(
                        ("Target volatility", f"{summary['TargetVol']:.2%}")
                    )

                cons_rows.append(("Return window", summary["Window"]))

                st.table(
                    styled_table(
                        pd.DataFrame(cons_rows, columns=["Constraint", "Setting"])
                    )
                )

        # ---------------- Current vs optimized table -----------------
        comp_pct = comp.copy()
        comp_pct["Current (%)"] = (100 * comp_pct["CurrentWeight"]).round(2)
        comp_pct["Optimized (%)"] = (100 * comp_pct["OptimizedWeight"]).round(2)
        comp_pct["Δ (%)"] = (100 * comp_pct["Change"]).round(2)

        # Fund display name
        if "FundName" in comp_pct.columns:
            comp_pct["Fund"] = comp_pct["FundName"]
        else:
            comp_pct["Fund"] = comp_pct.index.map(
                lambda t: TICKER_META.get(t, {}).get("name", t)
            )

        comp_pct = comp_pct.reset_index().rename(columns={"index": "Ticker"})
        comp_pct = comp_pct[
            ["Fund", "Ticker", "AssetGroup", "Current (%)", "Optimized (%)", "Δ (%)"]
        ]
        comp_pct = comp_pct.rename(columns={"AssetGroup": "Asset group"})

        # Hide rows that are basically zero in both portfolios
        threshold = 0.01
        mask = (comp_pct["Current (%)"].abs() > threshold) | (
            comp_pct["Optimized (%)"].abs() > threshold
        )
        comp_pct = comp_pct[mask]
        comp_pct = comp_pct.sort_values("Optimized (%)", ascending=False)

        st.markdown("**Current vs optimized weights (by fund)**")
        st.table(styled_table(comp_pct))

        # ---------------- Allocation by asset group ------------------
        st.markdown("**Allocation by asset group – current vs optimized**")
        fig_opt_bar = make_group_bar_figure(
            comp,
            current_col="CurrentWeight",
            optimized_col="OptimizedWeight",
        )
        st.plotly_chart(fig_opt_bar, use_container_width=True)

        # ---------------- Risk contribution by asset group -----------
        st.markdown("**Risk contribution by asset group (optimized portfolio)**")
        rc_df = compute_group_risk_contrib(rets_win, weights, comp)

        fig_rc_pie = make_risk_contrib_pie(rc_df)
        st.plotly_chart(fig_rc_pie, use_container_width=True)

        # -----------------------------------------------------------
        # Backtests (daily NAV) – 2 charts:
        #   1) Current vs static optimized
        #   2) Current vs rolling optimized
        # -----------------------------------------------------------
        st.markdown(f"**Historical backtest – {freq_label} rebalancing**")

        max_months = len(rets_win)

        # Rolling strategy needs a trailing window; can't start until we have
        # `lookback_months` of history.
        rolling_lookback = min(lookback_months, max_months)
        start_months_rolling = rolling_lookback

        # Static strategies (current, static optimized, 60/40) use the full
        # user-selected window (full vs last5). We can rebalance from the very
        # first month in the selected history.
        start_months_static = 1  # first monthly observation

        # Current portfolio weights from case (via analytics),
        # aligned to the return panel
        current_w_vec = (
            base_comp["CurrentWeight"]
            .reindex(rets_win.columns)
            .fillna(0.0)
        )

        # Policy 60/40 (MSCI World / US Agg)
        w_policy = make_policy_60_40(rets_win)

        # 60/40 over full static window
        bt_policy_static = backtest_quarterly_daily(
            rets_win,
            mode="static",
            lookback_months=None,  # not used when override_weights is set
            start_months=start_months_static,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=w_policy,
        )

        # 60/40 over rolling horizon only (for rolling chart)
        bt_policy_rolling = backtest_quarterly_daily(
            rets_win,
            mode="static",
            lookback_months=rolling_lookback,
            start_months=start_months_rolling,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=w_policy,
        )

        # (A) Current portfolio – static horizon (full selected window)
        bt_current_static = backtest_quarterly_daily(
            rets_win,
            mode="static",  # static, but with override_weights = current
            lookback_months=None,
            start_months=start_months_static,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=turnover_budget if turnover_budget > 0 else None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=current_w_vec,
        )

        # (A2) Current portfolio – rolling horizon only (for rolling chart)
        bt_current_rolling = backtest_quarterly_daily(
            rets_win,
            mode="static",
            lookback_months=rolling_lookback,
            start_months=start_months_rolling,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=turnover_budget if turnover_budget > 0 else None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=current_w_vec,
        )

        # (B) Static optimized – optimize once on full window
        bt_static = backtest_quarterly_daily(
            rets_win,
            mode="static",
            lookback_months=None,  # window logic handled inside
            start_months=start_months_static,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=turnover_budget if turnover_budget > 0 else None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=weights,
        )

        # (C) Rolling optimized – re-optimize every rebalance on trailing window
        bt_rolling = backtest_quarterly_daily(
            rets_win,
            mode="rolling",
            lookback_months=rolling_lookback,
            start_months=start_months_rolling,
            rebalance_freq=rebalance_freq,
            risk_profile=risk_profile,
            objective=objective,
            equity_cap=None,
            illiquid_cap=None,
            turnover_budget=None,
            include_real_assets=include_real_assets,
            include_non_us_equity=include_non_us_equity,
            include_short_duration_credit=include_short_credit,
            override_weights=None,
        )

        # -------- Chart 1: Current vs static optimized ---------------
        nav_df_static = pd.concat(
            [
                bt_current_static["nav"].rename("Current"),
                bt_static["nav"].rename("Static optimized"),
                bt_policy_static["nav"].rename("60/40 Benchmark (MSCI World / US Agg)"),
            ],
            axis=1,
        ).dropna(how="all")

        fig_static = px.line(
            nav_df_static,
            labels={
                "value": "Wealth",
                "variable": "Strategy",
                "index": "Date",
            },
        )
        fig_static.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend_title="",
        )
        fig_static.update_yaxes(tickformat="$,.2f")
        fig_static.update_traces(
            hovertemplate=(
                "Strategy: %{fullData.name}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Wealth: %{y:$,.2f}<extra></extra>"
            )
        )

        st.markdown(
        "**Current vs Static Optimized "
        "(optimized once on the selected history)**"
        )

        st.plotly_chart(fig_static, use_container_width=True)

        # -------- Chart 2: Current vs rolling optimized --------------
        nav_df_rolling = pd.concat(
            [
                bt_current_rolling["nav"].rename("Current"),
                bt_rolling["nav"].rename("Rolling optimized"),
                bt_policy_rolling["nav"].rename("60/40 Benchmark (MSCI World / US Agg)"),
            ],
            axis=1,
        ).dropna(how="all")

        fig_rolling = px.line(
            nav_df_rolling,
            labels={
                "value": "Wealth",
                "variable": "Strategy",
                "index": "Date",
            },
        )
        fig_rolling.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            legend_title="",
        )
        fig_rolling.update_yaxes(tickformat="$,.2f")
        fig_rolling.update_traces(
            hovertemplate=(
                "Strategy: %{fullData.name}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Wealth: %{y:$,.2f}<extra></extra>"
            )
        )

        st.markdown(
            f"**Current vs Rolling Optimized "
            f"(re-optimized every rebalance using last {rolling_lookback} months)**"
        )

        st.plotly_chart(fig_rolling, use_container_width=True)



    else:
        st.info(
            "Adjust the controls in the sidebar and click **Run optimization** "
            "to generate recommended allocations."
        )


if __name__ == "__main__":
    main()

