"""
Screening logic: filters, theme tagging, and forensic quality overlay.
Takes DataFrames from data_layer and returns the merged, scored DataFrame.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Universe filter
# ---------------------------------------------------------------------------

def apply_universe_filter(
    universe: pd.DataFrame,
    adtv: pd.DataFrame,
    mkt_cap_min_bn: float = 15.0,
    mkt_cap_max_bn: float = 100.0,
    adtv_min_m: float = 50.0,     # CNY million
) -> pd.DataFrame:
    """
    Returns filtered universe with liquidity flags.
    Excludes: ST stocks, negative equity (caught later), suspended names.
    """
    df = universe.copy()

    # Join ADTV
    if not adtv.empty:
        df = df.merge(adtv[["ticker", "adtv_60d"]], on="ticker", how="left")
    else:
        df["adtv_60d"] = np.nan

    # Market cap filter (CNY bn)
    if "mkt_cap_bn" in df.columns:
        mc_ok = df["mkt_cap_bn"].between(mkt_cap_min_bn, mkt_cap_max_bn, inclusive="both")
    else:
        mc_ok = pd.Series(True, index=df.index)

    # Exclude ST
    st_ok = ~df.get("is_st", pd.Series(False, index=df.index))

    # Liquidity
    adtv_m = df["adtv_60d"] / 1e6 if "adtv_60d" in df.columns else pd.Series(np.nan, index=df.index)
    adtv_ok   = adtv_m >= adtv_min_m
    adtv_warn = (adtv_m >= 30) & (adtv_m < adtv_min_m)
    adtv_red  = adtv_m < 30

    df["adtv_m"] = adtv_m
    df["liquidity_flag"] = "ok"
    df.loc[adtv_warn, "liquidity_flag"] = "warn"
    df.loc[adtv_red,  "liquidity_flag"] = "red"

    # Primary filter: market cap + not ST
    df["passes_filter"] = mc_ok & st_ok
    # ADTV: filter out red (<30m) and below floor, but keep 'warn' band visible
    df["passes_adtv"] = adtv_ok | adtv_warn  # include warn-band for visibility

    return df


# ---------------------------------------------------------------------------
# Theme tagging
# ---------------------------------------------------------------------------

def tag_themes(
    df: pd.DataFrame,
    financials: pd.DataFrame,
    cashflow: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds boolean theme columns and 0-100 fit scores.
    Theme 1: High dividend / shareholder return
    Theme 2: Self-sufficiency / localization
    Theme 3: Liquidity-driven growth / AI
    Theme 4: Quality consumer (pricing power)
    """
    merged = df.copy()
    if not financials.empty:
        merged = merged.merge(financials, on="ticker", how="left")
    if not cashflow.empty:
        merged = merged.merge(cashflow, on="ticker", how="left", suffixes=("", "_cf"))
    if not industry.empty:
        merged = merged.merge(industry[["ticker","industry_boards","is_localization"]],
                               on="ticker", how="left")

    # ---- Theme 1: High dividend / shareholder return ----------------------
    # Use div_yield from universe spot data; payout from cashflow if available
    dy = pd.to_numeric(merged.get("div_yield", pd.Series(np.nan, index=merged.index)),
                       errors="coerce")

    # Sector-median dividend yield for relative comparison
    if "div_yield" in merged.columns and "sector" in merged.columns:
        sector_median = merged.groupby("sector")["div_yield"].transform("median")
        dy_above_median = dy > sector_median
    else:
        dy_above_median = dy > 2.0  # fallback: absolute 2% floor

    # Payout ratio proxy: not easily available from abstract; skip for now
    payout_ok = pd.Series(True, index=merged.index)  # conservative default
    div_3y_growth = merged.get("div_3y_growth", pd.Series(False, index=merged.index)).fillna(False)
    fcf_pos = merged.get("fcf_positive_3y", pd.Series(False, index=merged.index)).fillna(False)

    merged["theme_dividend"] = (
        dy_above_median & payout_ok & div_3y_growth.astype(bool) & fcf_pos.astype(bool)
    )
    # Score: 25 per criterion
    merged["score_dividend"] = (
        dy_above_median.astype(int) * 25
        + payout_ok.astype(int) * 25
        + div_3y_growth.astype(bool).astype(int) * 25
        + fcf_pos.astype(bool).astype(int) * 25
    )

    # ---- Theme 2: Self-sufficiency / localization --------------------------
    merged["theme_localization"] = merged.get(
        "is_localization", pd.Series(False, index=merged.index)
    ).fillna(False).astype(bool)
    merged["score_localization"] = merged["theme_localization"].astype(int) * 100

    # ---- Theme 3: Liquidity-driven growth / AI ----------------------------
    rev_g = pd.to_numeric(
        merged.get("rev_growth", pd.Series(np.nan, index=merged.index)), errors="coerce"
    )
    high_rev_growth = rev_g > 15.0  # >15% YoY
    # Positive but volatile earnings: net_margin positive
    nm = pd.to_numeric(
        merged.get("net_margin", pd.Series(np.nan, index=merged.index)), errors="coerce"
    )
    pos_earnings = nm > 0

    merged["theme_ai_growth"] = (high_rev_growth & pos_earnings)
    merged["score_ai_growth"] = (
        (rev_g.clip(0, 50) / 50 * 60).fillna(0)
        + pos_earnings.astype(int) * 40
    ).clip(0, 100)

    # ---- Theme 4: Quality consumer (pricing power) -------------------------
    gm = pd.to_numeric(
        merged.get("gross_margin", pd.Series(np.nan, index=merged.index)), errors="coerce"
    )
    gm_std = pd.to_numeric(
        merged.get("gross_margin_std", pd.Series(np.nan, index=merged.index)), errors="coerce"
    )
    gm_stable  = gm_std < 3.0   # std dev < 3pp
    gm_high    = gm > 35.0       # >35%
    rev_pos    = rev_g > 0

    merged["theme_quality_consumer"] = (gm_stable & gm_high & rev_pos)
    merged["score_quality_consumer"] = (
        gm_stable.astype(int) * 30
        + (gm.clip(35, 70) - 35).fillna(0) / 35 * 40
        + rev_pos.astype(int) * 30
    ).clip(0, 100)

    return merged


# ---------------------------------------------------------------------------
# Forensic quality overlay
# ---------------------------------------------------------------------------

def apply_forensics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and flags forensic quality metrics.
    Adds columns: accruals_flag, cc_flag, rec_flag, dupont_*
    """
    out = df.copy()

    # Accruals ratio flags
    ar = pd.to_numeric(out.get("accruals_ratio", pd.Series(np.nan, index=out.index)),
                       errors="coerce")
    out["accruals_flag"] = "green"
    out.loc[ar.abs() > 0.03, "accruals_flag"] = "yellow"
    out.loc[ar > 0.05,        "accruals_flag"] = "red"
    # Negative accruals (OCF > NI) is actually good sign
    out.loc[ar < -0.05,       "accruals_flag"] = "green"

    # Cash conversion flag
    cc = pd.to_numeric(out.get("cash_conversion", pd.Series(np.nan, index=out.index)),
                       errors="coerce")
    cc_red = out.get("cc_red_flag", pd.Series(False, index=out.index)).fillna(False)
    out["cc_flag"] = "green"
    out.loc[cc < 0.7, "cc_flag"] = "yellow"
    out.loc[cc_red.astype(bool), "cc_flag"] = "red"

    # Receivables flag
    rec_flag = out.get("rec_vs_rev_flag", pd.Series(False, index=out.index)).fillna(False)
    out["rec_flag"] = out.apply(
        lambda r: "red" if rec_flag[r.name] else "green", axis=1
    )

    # DuPont decomposition: ROE = net_margin × asset_turnover × equity_multiplier
    nm = pd.to_numeric(out.get("net_margin", pd.Series(np.nan, index=out.index)), errors="coerce")
    at = pd.to_numeric(out.get("asset_turnover", pd.Series(np.nan, index=out.index)), errors="coerce")
    de = pd.to_numeric(out.get("debt_to_equity", pd.Series(np.nan, index=out.index)), errors="coerce")
    # equity_multiplier ≈ 1 / (1 - D/A) where D/A = debt_to_equity (expressed as %)
    em = 1 / (1 - (de / 100).clip(0, 0.95))  # cap leverage

    out["dupont_margin"]   = nm
    out["dupont_turnover"] = at
    out["dupont_leverage"] = em
    out["dupont_roe_check"] = nm * at * em  # should approximate stated ROE

    # ROE driven by leverage: flag if leverage > 3x but margin is mediocre
    out["leverage_driven_roe"] = (em > 3) & (nm < 10)

    return out


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def build_screen(
    universe: pd.DataFrame,
    adtv: pd.DataFrame,
    financials: pd.DataFrame,
    cashflow: pd.DataFrame,
    industry: pd.DataFrame,
    valuation: pd.DataFrame,
    mkt_cap_min_bn: float = 15.0,
    mkt_cap_max_bn: float = 100.0,
    adtv_min_m: float = 50.0,
) -> pd.DataFrame:
    """
    Master function: merge everything, filter, tag, and flag.
    Returns full DataFrame; caller can filter passes_filter.
    """
    df = apply_universe_filter(universe, adtv, mkt_cap_min_bn, mkt_cap_max_bn, adtv_min_m)
    df = tag_themes(df, financials, cashflow, industry)
    df = apply_forensics(df)

    if not valuation.empty:
        df = df.merge(valuation, on="ticker", how="left")

    # 1Y price change — use pct_ytd if 60d not available
    if "pct_ytd" in df.columns:
        df["price_1y_chg"] = df["pct_ytd"]
    elif "pct_60d" in df.columns:
        df["price_1y_chg"] = df["pct_60d"]

    # Composite quality score: 0-100 (higher = better quality)
    good = ((df["accruals_flag"] == "green").astype(int) * 30
            + (df["cc_flag"] != "red").astype(int) * 30
            + (df["rec_flag"] != "red").astype(int) * 20
            + (~df.get("leverage_driven_roe", pd.Series(False, index=df.index)).fillna(False)).astype(int) * 20)
    df["quality_score"] = good

    return df


def display_columns() -> list[str]:
    """Ordered columns for the main table view."""
    return [
        "ticker", "name", "sector", "mkt_cap_bn", "adtv_m",
        "div_yield", "pe_ttm", "pb",
        "theme_dividend", "theme_localization", "theme_ai_growth", "theme_quality_consumer",
        "accruals_ratio", "accruals_flag",
        "cash_conversion", "cc_flag",
        "roe", "gross_margin", "net_margin",
        "dupont_margin", "dupont_turnover", "dupont_leverage",
        "quality_score",
        "price_1y_chg",
        "liquidity_flag",
    ]
