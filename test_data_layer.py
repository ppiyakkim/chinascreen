"""
Step-by-step data layer validation — run before the UI to confirm logic is correct.
Usage: python test_data_layer.py
"""
import sys
import logging
import jsonpath_ng
if "jsonpath" not in sys.modules:
    sys.modules["jsonpath"] = jsonpath_ng

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd
import numpy as np
from data_layer import (
    fetch_universe, fetch_adtv, fetch_financials_batch,
    fetch_cashflow_batch, fetch_valuation_percentile,
    fetch_industry_tags, fetch_dividend_yield,
)
from screener import build_screen

SAMPLE_N = 20

print("=" * 60)
print("STEP 1: Universe fetch")
print("=" * 60)
universe = fetch_universe(force=False)
print(f"Total stocks: {len(universe)}")
print(f"Columns: {list(universe.columns)}")
print(f"\nSample:\n{universe[['ticker','name','mkt_cap_bn','pe_ttm','pb']].head(10).to_string()}")

if "mkt_cap_bn" in universe.columns:
    mid_cap = universe[universe["mkt_cap_bn"].between(15, 100)]
    print(f"\nMid-cap band (15-100bn): {len(mid_cap)} stocks")
    print(f"ST stocks excluded: {universe['is_st'].sum()}")

print("\n" + "=" * 60)
print("STEP 2: Mid-cap pre-filter, take 20-stock sample")
print("=" * 60)
sample_tickers = universe[
    universe["mkt_cap_bn"].between(10, 120)
]["ticker"].dropna().astype(str).tolist()[:SAMPLE_N]
print(f"Sample tickers: {sample_tickers}")

print("\n" + "=" * 60)
print("STEP 3: ADTV")
print("=" * 60)
adtv = fetch_adtv(sample_tickers, force=False)
print(f"ADTV rows: {len(adtv)}")
print(adtv.to_string())

print("\n" + "=" * 60)
print("STEP 4: Financials")
print("=" * 60)
fins = fetch_financials_batch(sample_tickers, force=False)
print(f"Financials rows: {len(fins)}")
fin_cols = [c for c in ["ticker","roe","gross_margin","net_margin","asset_turnover","rev_growth"] if c in fins.columns]
print(fins[fin_cols].to_string())

print("\n" + "=" * 60)
print("STEP 5: Cash-flow / accruals")
print("=" * 60)
cf = fetch_cashflow_batch(sample_tickers, force=False)
print(f"Cashflow rows: {len(cf)}")
cf_cols = [c for c in ["ticker","accruals_ratio","cash_conversion","cc_red_flag","fcf_positive_3y"] if c in cf.columns]
print(cf[cf_cols].to_string())

print("\n" + "=" * 60)
print("STEP 6: Valuation percentile")
print("=" * 60)
val = fetch_valuation_percentile(sample_tickers[:5], force=False)  # just 5 to save time
print(val.to_string())

print("\n" + "=" * 60)
print("STEP 7: Industry tags")
print("=" * 60)
ind = fetch_industry_tags(sample_tickers, force=False)
print(ind[ind["is_localization"]].to_string())

print("\n" + "=" * 60)
print("STEP 8: Full screen assembly")
print("=" * 60)
sample_universe = universe[universe["ticker"].isin(sample_tickers)].copy()
div_df = fetch_dividend_yield(sample_universe)
sample_universe = sample_universe.merge(div_df, on="ticker", how="left")

screen = build_screen(
    universe=sample_universe,
    adtv=adtv,
    financials=fins,
    cashflow=cf,
    industry=ind,
    valuation=val,
    mkt_cap_min_bn=15.0,
    mkt_cap_max_bn=100.0,
    adtv_min_m=50.0,
)

print(f"\nScreen output: {len(screen)} rows")
print(f"Passes filter: {screen['passes_filter'].sum()}")
print(f"Theme - Dividend: {screen['theme_dividend'].sum()}")
print(f"Theme - Localization: {screen['theme_localization'].sum()}")
print(f"Theme - AI/Growth: {screen['theme_ai_growth'].sum()}")
print(f"Theme - Quality Consumer: {screen['theme_quality_consumer'].sum()}")
print(f"\nAccruals flags:\n{screen['accruals_flag'].value_counts().to_string()}")
print(f"\nCC flags:\n{screen['cc_flag'].value_counts().to_string()}")

# Key columns
cols = [c for c in [
    "ticker","name","mkt_cap_bn","adtv_m",
    "theme_dividend","theme_localization","theme_ai_growth","theme_quality_consumer",
    "accruals_ratio","accruals_flag","cash_conversion","cc_flag",
    "roe","gross_margin","quality_score",
] if c in screen.columns]
print(f"\n{'='*60}")
print("FINAL SCREEN TABLE (sample)")
print("=" * 60)
print(screen[cols].to_string())
