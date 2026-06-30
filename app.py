"""
China A-Share Mid-Cap Screener Dashboard
Run locally: streamlit run app.py
Demo mode (no network):  DEMO_MODE=1 streamlit run app.py
"""
import os
import sys
import logging

# jsonpath shim must happen before akshare import anywhere in the process
import jsonpath_ng
if "jsonpath" not in sys.modules:
    sys.modules["jsonpath"] = jsonpath_ng

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

if not DEMO_MODE:
    from data_layer import (
        fetch_universe, fetch_adtv, fetch_financials_batch,
        fetch_cashflow_batch, fetch_valuation_percentile,
        fetch_industry_tags, fetch_dividend_yield, cache_status,
    )
else:
    from demo_data import get_all_demo_data

from screener import build_screen, display_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="China A-Share Mid-Cap Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("China A-Share Mid-Cap Screener")
if DEMO_MODE:
    st.warning("DEMO MODE — using synthetic data. Run without DEMO_MODE=1 for live akshare data.")
st.caption("Data: akshare · Cache: local SQLite · Themes: Schroders-aligned · Forensics: Penman-style")

# ============================================================
# Session-state init
# ============================================================
if "screen_df" not in st.session_state:
    st.session_state.screen_df = pd.DataFrame()
if "exclusion_list" not in st.session_state:
    st.session_state.exclusion_list = []

# ============================================================
# Sidebar — filters
# ============================================================
with st.sidebar:
    st.header("Filters")

    mkt_cap_range = st.slider(
        "Market Cap (CNY bn)", min_value=5.0, max_value=300.0,
        value=(15.0, 100.0), step=5.0,
    )
    adtv_floor = st.slider(
        "Min ADTV (CNY m)", min_value=10.0, max_value=500.0,
        value=50.0, step=10.0,
    )

    st.subheader("Theme filter")
    theme_filter = st.multiselect(
        "Show stocks tagged with (any of):",
        options=["High Dividend", "Localization/Self-Sufficiency",
                 "AI/Growth", "Quality Consumer"],
        default=[],
    )

    st.subheader("Quality flags")
    hide_red_accruals = st.checkbox("Hide red accruals (>5%)", value=False)
    hide_red_cc       = st.checkbox("Hide red cash conversion", value=False)

    st.subheader("Exclusion list")
    st.caption("Paste tickers confirmed held by Schroders (from semi-annual PDF disclosures)")
    excl_text = st.text_area(
        "One ticker per line:",
        value="\n".join(st.session_state.exclusion_list),
        height=120,
    )
    if st.button("Update exclusion list"):
        st.session_state.exclusion_list = [
            t.strip() for t in excl_text.split("\n") if t.strip()
        ]
        st.success(f"Updated: {len(st.session_state.exclusion_list)} tickers")

# ============================================================
# Data load controls
# ============================================================
if not DEMO_MODE:
    col_refresh, col_sample, col_status = st.columns([1, 1, 2])
    with col_refresh:
        do_refresh = st.button(
            "Refresh cache", type="primary",
            help="Re-fetches all data from akshare. Full universe: ~30-60 min."
        )
    with col_sample:
        sample_mode = st.checkbox(
            "Sample mode (20 stocks)", value=True,
            help="Quick validation with 20 stocks before full run"
        )
    with col_status:
        st.caption("Cache status")
        status = cache_status()
        for tbl, info in status.items():
            color = "green" if info["rows"] > 0 else "red"
            ts = info["fetched_at"][:16] if info["fetched_at"] != "never" else "never"
            st.caption(f":{color}[{tbl}]: {info['rows']} rows · {ts}")
else:
    do_refresh = st.button("Regenerate demo data", type="primary")
    sample_mode = False
    col_status = st.empty()

# ============================================================
# Data pipeline
# ============================================================

def run_demo_pipeline():
    data = get_all_demo_data()
    universe = data["universe"]
    div_df = universe[["ticker", "div_yield"]]
    screen = build_screen(
        universe=universe,
        adtv=data["adtv"],
        financials=data["financials"],
        cashflow=data["cashflow"],
        industry=data["industry"],
        valuation=data["valuation"],
        mkt_cap_min_bn=mkt_cap_range[0],
        mkt_cap_max_bn=mkt_cap_range[1],
        adtv_min_m=adtv_floor,
    )
    st.session_state.screen_df = screen
    st.success(f"Demo data loaded — {len(screen)} stocks, {screen['passes_filter'].sum()} pass filter.")


def run_live_pipeline(force: bool = False, sample: bool = True):
    with st.spinner("Loading universe …"):
        try:
            universe = fetch_universe(force=force)
        except Exception as e:
            st.error(f"Failed to fetch universe: {e}\n\nTip: run with DEMO_MODE=1 to test the UI.")
            return

        if universe.empty:
            st.error("Universe is empty.")
            return

        if "mkt_cap_bn" in universe.columns:
            mc_pre = universe[
                universe["mkt_cap_bn"].between(
                    mkt_cap_range[0] * 0.8, mkt_cap_range[1] * 1.2
                )
            ]
        else:
            mc_pre = universe

        tickers = mc_pre["ticker"].dropna().astype(str).tolist()
        if sample:
            tickers = tickers[:20]
            mc_pre = mc_pre[mc_pre["ticker"].isin(tickers)]

    st.info(f"Working on {len(tickers)} tickers …")

    with st.spinner("ADTV (60-day history) …"):
        adtv = fetch_adtv(tickers, force=force)

    with st.spinner("Financials …"):
        fins = fetch_financials_batch(tickers, force=force)

    with st.spinner("Cash-flow / accruals …"):
        cf = fetch_cashflow_batch(tickers, force=force)

    with st.spinner("Valuation percentiles …"):
        val = fetch_valuation_percentile(tickers, force=force)

    with st.spinner("Industry tags (localization) …"):
        ind = fetch_industry_tags(tickers, force=force)

    with st.spinner("Assembling screen …"):
        div_df = fetch_dividend_yield(mc_pre)
        mc_with_div = mc_pre.merge(div_df, on="ticker", how="left")
        screen = build_screen(
            universe=mc_with_div,
            adtv=adtv,
            financials=fins,
            cashflow=cf,
            industry=ind,
            valuation=val,
            mkt_cap_min_bn=mkt_cap_range[0],
            mkt_cap_max_bn=mkt_cap_range[1],
            adtv_min_m=adtv_floor,
        )

    st.session_state.screen_df = screen
    st.success(f"Done — {len(screen)} rows, {screen['passes_filter'].sum()} pass primary filter.")


if do_refresh or st.session_state.screen_df.empty:
    if DEMO_MODE:
        run_demo_pipeline()
    else:
        run_live_pipeline(force=do_refresh, sample=sample_mode)

# ============================================================
# Display
# ============================================================
df = st.session_state.screen_df

if df.empty:
    st.info("Click 'Refresh cache' to load data.")
    st.stop()

# Re-apply filter sliders dynamically (so slider changes don't need re-fetch)
if "mkt_cap_bn" in df.columns:
    mc_mask = df["mkt_cap_bn"].between(mkt_cap_range[0], mkt_cap_range[1])
else:
    mc_mask = pd.Series(True, index=df.index)

if "is_st" in df.columns:
    st_mask = ~df["is_st"]
else:
    st_mask = pd.Series(True, index=df.index)

# ADTV: exclude below 30m, warn for 30-floor, include all above floor
adtv_m = df.get("adtv_m", pd.Series(np.nan, index=df.index))
adtv_mask = adtv_m.isna() | (adtv_m >= 30)  # keep NaN (no data) and >=30m

view = df[mc_mask & st_mask & adtv_mask].copy()

# Recompute ADTV flag with current slider value
if "adtv_m" in view.columns:
    view["liquidity_flag"] = "ok"
    view.loc[view["adtv_m"].between(30, adtv_floor, inclusive="left"), "liquidity_flag"] = "warn"
    view.loc[view["adtv_m"] < 30, "liquidity_flag"] = "red"

# Theme filter
theme_col_map = {
    "High Dividend": "theme_dividend",
    "Localization/Self-Sufficiency": "theme_localization",
    "AI/Growth": "theme_ai_growth",
    "Quality Consumer": "theme_quality_consumer",
}
if theme_filter:
    mask = pd.Series(False, index=view.index)
    for t in theme_filter:
        col = theme_col_map.get(t)
        if col and col in view.columns:
            mask = mask | view[col].fillna(False).astype(bool)
    view = view[mask]

if hide_red_accruals and "accruals_flag" in view.columns:
    view = view[view["accruals_flag"] != "red"]
if hide_red_cc and "cc_flag" in view.columns:
    view = view[view["cc_flag"] != "red"]

# Mark exclusion list
excl = set(st.session_state.exclusion_list)
view["excluded"] = view["ticker"].isin(excl)

# ---- Summary metrics ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks in view", len(view))
c2.metric("High Dividend", int(view.get("theme_dividend", pd.Series(False)).fillna(False).sum()))
c3.metric("Localization", int(view.get("theme_localization", pd.Series(False)).fillna(False).sum()))
c4.metric("AI/Growth", int(view.get("theme_ai_growth", pd.Series(False)).fillna(False).sum()))
c5.metric("Quality Consumer", int(view.get("theme_quality_consumer", pd.Series(False)).fillna(False).sum()))

st.markdown("---")

# ============================================================
# Build display table
# ============================================================
DISPLAY_COLS = [c for c in display_columns() if c in view.columns]
table_df = view[DISPLAY_COLS + ["excluded"]].copy()

COL_LABELS = {
    "ticker": "Ticker", "name": "Name", "sector": "Sector",
    "mkt_cap_bn": "Mkt Cap (bn¥)", "adtv_m": "ADTV (m¥)",
    "div_yield": "Div Yield %", "pe_ttm": "PE (TTM)", "pb": "PB",
    "theme_dividend": "Div Theme", "theme_localization": "Local Theme",
    "theme_ai_growth": "AI/Growth", "theme_quality_consumer": "Quality Theme",
    "accruals_ratio": "Accruals", "accruals_flag": "⚑ Accruals",
    "cash_conversion": "Cash Conv", "cc_flag": "⚑ CC",
    "roe": "ROE %", "gross_margin": "Gross Margin %", "net_margin": "Net Margin %",
    "dupont_margin": "DuPont: Margin", "dupont_turnover": "DuPont: Turns",
    "dupont_leverage": "DuPont: Leverage",
    "quality_score": "Quality Score", "price_1y_chg": "YTD %",
    "liquidity_flag": "⚑ Liquidity", "excluded": "Held (excl.)",
}
table_df = table_df.rename(columns=COL_LABELS)


def flag_color(val):
    if val == "red":
        return "background-color: #ff4b4b; color: white"
    if val == "yellow":
        return "background-color: #ffd700"
    if val == "green":
        return "background-color: #21c354; color: white"
    if val == "warn":
        return "background-color: #ffd700"
    if val == "ok":
        return "background-color: #21c354; color: white"
    return ""


def excluded_style(val):
    return "color: #aaaaaa; text-decoration: line-through" if val else ""


def bool_display(val):
    try:
        if pd.isna(val):
            return ""
    except:
        pass
    return "✓" if val else ""


for col in ["Div Theme", "Local Theme", "AI/Growth", "Quality Theme", "Held (excl.)"]:
    if col in table_df.columns:
        table_df[col] = table_df[col].apply(bool_display)

# Round numerics
num_cols = ["Mkt Cap (bn¥)", "ADTV (m¥)", "Div Yield %", "PE (TTM)", "PB",
            "Accruals", "Cash Conv", "ROE %", "Gross Margin %", "Net Margin %",
            "DuPont: Margin", "DuPont: Turns", "DuPont: Leverage", "YTD %", "Quality Score"]
for col in num_cols:
    if col in table_df.columns:
        table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(2)

flag_cols = [c for c in ["⚑ Accruals", "⚑ CC", "⚑ Liquidity"] if c in table_df.columns]
held_col  = ["Held (excl.)"] if "Held (excl.)" in table_df.columns else []

styled = (
    table_df.style
    .applymap(flag_color, subset=flag_cols)
    .applymap(excluded_style, subset=held_col)
    .format(na_rep="—")
)

st.dataframe(styled, use_container_width=True, height=520)

# ============================================================
# Export
# ============================================================
non_excl = view[~view.get("excluded", pd.Series(False, index=view.index)).fillna(False)]
csv_bytes = non_excl[DISPLAY_COLS].to_csv(index=False).encode()
st.download_button(
    "⬇ Export non-excluded to CSV",
    data=csv_bytes,
    file_name="china_ashare_screen.csv",
    mime="text/csv",
)

# ============================================================
# Detail panel
# ============================================================
st.markdown("---")
st.subheader("Stock Detail Panel")

ticker_list = view["ticker"].dropna().astype(str).tolist()
selected_ticker = st.selectbox("Select ticker:", options=[""] + ticker_list)

if selected_ticker:
    rows = view[view["ticker"] == selected_ticker]
    if rows.empty:
        st.warning("Not found.")
    else:
        row = rows.iloc[0]

        # Header
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Name", str(row.get("name", "—")))
        col_b.metric("Sector", str(row.get("sector", "—")))
        col_c.metric("Mkt Cap", f"CNY {row.get('mkt_cap_bn', 0):.1f}bn")
        col_d.metric("ADTV", f"CNY {row.get('adtv_m', 0):.0f}m")

        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("ROE", f"{row.get('roe', 0):.1f}%")
        col_f.metric("Gross Margin", f"{row.get('gross_margin', 0):.1f}%")
        col_g.metric("Accruals Ratio", f"{row.get('accruals_ratio', 0):.3f}",
                     delta="High ⚠" if row.get("accruals_flag") == "red" else "OK",
                     delta_color="inverse" if row.get("accruals_flag") == "red" else "normal")
        col_h.metric("Cash Conv", f"{row.get('cash_conversion', 0):.2f}")

        # Themes
        tags = []
        for label, col in [
            ("High Dividend", "theme_dividend"),
            ("Localization", "theme_localization"),
            ("AI/Growth", "theme_ai_growth"),
            ("Quality Consumer", "theme_quality_consumer"),
        ]:
            if row.get(col):
                tags.append(f"`{label}`")
        st.markdown("**Themes:** " + (" · ".join(tags) if tags else "_none matched_"))

        if row.get("excluded"):
            st.error("This ticker is in your exclusion list (already held).")

        # Industry
        if row.get("industry_boards"):
            st.caption(f"Industry boards: {row['industry_boards']}")

        tab_dupont, tab_forensic, tab_valuation = st.tabs(
            ["DuPont Decomposition", "Forensic Quality", "Valuation Percentile"]
        )

        with tab_dupont:
            dm  = row.get("dupont_margin", np.nan)
            dt  = row.get("dupont_turnover", np.nan)
            dl  = row.get("dupont_leverage", np.nan)
            drc = row.get("dupont_roe_check", np.nan)
            stated_roe = row.get("roe", np.nan)

            fig_dp = go.Figure()
            components = ["Net Margin", "× Asset Turnover", "× Equity Multiplier", "= Implied ROE", "Stated ROE"]
            values = [dm, dt, dl, drc, stated_roe]
            colors = ["steelblue", "steelblue", "steelblue",
                      "green" if not pd.isna(drc) and drc > 0 else "red", "gray"]
            fig_dp.add_trace(go.Bar(
                x=components,
                y=[v if not pd.isna(v) else 0 for v in values],
                marker_color=colors,
                text=[f"{v:.2f}" if not pd.isna(v) else "—" for v in values],
                textposition="outside",
            ))
            fig_dp.update_layout(
                title="DuPont ROE Breakdown",
                yaxis_title="Value (% or ×)",
                height=350,
            )
            st.plotly_chart(fig_dp, use_container_width=True)

            if row.get("leverage_driven_roe"):
                st.warning("⚠ ROE appears leverage-driven (EM >3× but net margin <10%)")

        with tab_forensic:
            forensic_rows = [
                {
                    "Metric": "Accruals Ratio (NI−OCF)/TA",
                    "Value": f"{row.get('accruals_ratio', np.nan):.3f}" if not pd.isna(row.get("accruals_ratio", np.nan)) else "—",
                    "Flag": row.get("accruals_flag", "—"),
                    "Interpretation": ">5% red: earnings may not be backed by cash. Negative is good.",
                },
                {
                    "Metric": "Cash Conversion (OCF/NI avg 4yr)",
                    "Value": f"{row.get('cash_conversion', np.nan):.2f}" if not pd.isna(row.get("cash_conversion", np.nan)) else "—",
                    "Flag": row.get("cc_flag", "—"),
                    "Interpretation": "<0.7 for 2 consecutive years = red",
                },
                {
                    "Metric": "Receivables growth vs Revenue growth",
                    "Value": "Warn" if row.get("rec_vs_rev_flag") else "OK",
                    "Flag": row.get("rec_flag", "—"),
                    "Interpretation": "Rec growth >1.5× rev growth suggests channel stuffing",
                },
                {
                    "Metric": "Leverage-driven ROE",
                    "Value": "Yes" if row.get("leverage_driven_roe") else "No",
                    "Flag": "red" if row.get("leverage_driven_roe") else "green",
                    "Interpretation": "EM >3× with net margin <10% = balance-sheet risk",
                },
            ]

            def flag_cell(val):
                if val == "red":    return "background-color:#ff4b4b;color:white"
                if val == "yellow": return "background-color:#ffd700"
                if val == "green":  return "background-color:#21c354;color:white"
                return ""

            f_df = pd.DataFrame(forensic_rows)
            st.dataframe(
                f_df.style.applymap(flag_cell, subset=["Flag"]),
                use_container_width=True,
                hide_index=True,
            )

        with tab_valuation:
            pe_pct = row.get("pe_pct", np.nan)
            pb_pct = row.get("pb_pct", np.nan)

            if not pd.isna(pe_pct) or not pd.isna(pb_pct):
                v1, v2 = st.columns(2)
                for label, pct, cur, lo, hi, col_target in [
                    ("PE Percentile (own 5yr history)",
                     pe_pct, row.get("pe_current"), row.get("pe_5y_low"), row.get("pe_5y_high"), v1),
                    ("PB Percentile (own 5yr history)",
                     pb_pct, row.get("pb_current"), None, None, v2),
                ]:
                    if pd.isna(pct):
                        col_target.caption(f"{label}: no data")
                        continue
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(pct, 1),
                        title={"text": label},
                        number={"suffix": "th pct"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "steelblue", "thickness": 0.3},
                            "steps": [
                                {"range": [0, 25],  "color": "#21c354"},
                                {"range": [25, 75], "color": "#ffd700"},
                                {"range": [75, 100],"color": "#ff4b4b"},
                            ],
                            "threshold": {
                                "line": {"color": "navy", "width": 3},
                                "thickness": 0.8,
                                "value": pct,
                            },
                        },
                    ))
                    fig_g.update_layout(height=280)
                    col_target.plotly_chart(fig_g, use_container_width=True)
                    if cur is not None and not pd.isna(cur):
                        col_target.caption(
                            f"Current: {cur:.1f} | "
                            + (f"5yr range: {lo:.1f}–{hi:.1f}" if lo is not None else "")
                        )
            else:
                st.info("No valuation history available for this ticker.")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "Screening logic based on Schroders China A positioning framework. "
    "Forensic quality overlay follows Penman (2013) accruals methodology. "
    "Exclusion list is manual — paste tickers from Schroders semi-annual PDF disclosures. "
    "Research tool only, not investment advice. Data: akshare (public endpoints)."
)
