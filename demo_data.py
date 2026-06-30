"""
Generates realistic mock data for UI testing when akshare endpoints are unreachable.
Used automatically when DEMO_MODE=1 or when fetches fail in the sandbox.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SECTORS = ["信息技术", "工业", "消费品", "医疗健康", "材料", "金融", "能源"]
INDUSTRIES = [
    "半导体设备", "工业自动化", "机器人", "锂电池材料",
    "消费电子", "白酒", "医疗器械", "化工材料",
    "新能源", "精密制造", "金融科技", "储能",
]

N = 80  # mock universe size


def _ticker(i: int) -> str:
    prefix = "60" if i % 2 == 0 else "00"
    return f"{prefix}{i:04d}"


def make_universe() -> pd.DataFrame:
    tickers = [_ticker(i + 1) for i in range(N)]
    mkt_caps = RNG.uniform(8, 120, N)  # CNY bn
    rows = []
    for i, tk in enumerate(tickers):
        sec = SECTORS[i % len(SECTORS)]
        rows.append({
            "ticker": tk,
            "name": f"示例股份{i+1:02d}",
            "price": round(RNG.uniform(5, 150), 2),
            "pct_change": round(RNG.uniform(-5, 5), 2),
            "volume": int(RNG.uniform(1e6, 1e8)),
            "turnover": round(RNG.uniform(5e6, 5e8), 0),
            "mkt_cap": mkt_caps[i] * 1e8,
            "float_mkt_cap": mkt_caps[i] * 0.7 * 1e8,
            "mkt_cap_bn": mkt_caps[i],
            "pe_ttm": round(RNG.uniform(8, 60), 1),
            "pb": round(RNG.uniform(0.8, 8), 2),
            "pct_60d": round(RNG.uniform(-20, 30), 2),
            "pct_ytd": round(RNG.uniform(-25, 40), 2),
            "is_st": bool(RNG.random() < 0.03),
            "sector": sec,
            "div_yield": round(RNG.uniform(0, 6), 2),
        })
    return pd.DataFrame(rows)


def make_adtv(tickers: list[str]) -> pd.DataFrame:
    adtvs = RNG.uniform(10e6, 800e6, len(tickers))
    return pd.DataFrame({"ticker": tickers, "adtv_60d": adtvs})


def make_financials(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        gm = RNG.uniform(15, 70)
        rows.append({
            "ticker": tk,
            "roe": round(RNG.uniform(-5, 35), 2),
            "gross_margin": round(gm, 2),
            "net_margin": round(RNG.uniform(-2, 25), 2),
            "asset_turnover": round(RNG.uniform(0.3, 2.0), 3),
            "current_ratio": round(RNG.uniform(0.8, 3.5), 2),
            "debt_to_equity": round(RNG.uniform(10, 70), 1),
            "rev_growth": round(RNG.uniform(-10, 40), 2),
            "eps": round(RNG.uniform(0.1, 5.0), 2),
            "gross_margin_std": round(RNG.uniform(0.5, 6), 2),
        })
    return pd.DataFrame(rows)


def make_cashflow(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        ni = RNG.uniform(1e8, 20e8)
        ocf_ratio = RNG.uniform(0.3, 1.8)
        ocf = ni * ocf_ratio
        ta = ni * RNG.uniform(5, 25)
        rows.append({
            "ticker": tk,
            "accruals_ratio": round((ni - ocf) / ta, 4),
            "cash_conversion": round(ocf_ratio, 3),
            "cc_red_flag": bool(ocf_ratio < 0.7 and RNG.random() < 0.5),
            "rec_vs_rev_flag": bool(RNG.random() < 0.15),
            "rec_growth": round(RNG.uniform(-0.1, 0.4), 3),
            "rev_growth_cf": round(RNG.uniform(-0.05, 0.3), 3),
            "div_3y_growth": bool(RNG.random() < 0.6),
            "fcf_positive_3y": bool(RNG.random() < 0.7),
        })
    return pd.DataFrame(rows)


def make_valuation(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        pe_pct = RNG.uniform(5, 95)
        pb_pct = RNG.uniform(5, 95)
        rows.append({
            "ticker": tk,
            "pe_current": round(RNG.uniform(8, 60), 1),
            "pe_pct": round(pe_pct, 1),
            "pe_5y_low": round(RNG.uniform(5, 15), 1),
            "pe_5y_high": round(RNG.uniform(50, 100), 1),
            "pb_current": round(RNG.uniform(0.8, 8), 2),
            "pb_pct": round(pb_pct, 1),
        })
    return pd.DataFrame(rows)


def make_industry(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, tk in enumerate(tickers):
        ind = INDUSTRIES[i % len(INDUSTRIES)]
        is_loc = ind in ["半导体设备", "工业自动化", "机器人", "锂电池材料",
                          "新能源", "精密制造", "储能"]
        rows.append({
            "ticker": tk,
            "industry_boards": ind,
            "is_localization": is_loc,
        })
    return pd.DataFrame(rows)


def get_all_demo_data() -> dict:
    universe = make_universe()
    tickers = universe["ticker"].tolist()
    return {
        "universe": universe,
        "adtv": make_adtv(tickers),
        "financials": make_financials(tickers),
        "cashflow": make_cashflow(tickers),
        "valuation": make_valuation(tickers),
        "industry": make_industry(tickers),
    }
