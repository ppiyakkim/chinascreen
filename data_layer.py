"""
Data layer: akshare fetchers with SQLite caching.
All network calls live here; app.py only calls these functions.
"""
import os
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# shim needed before akshare import
import jsonpath_ng
import sys
if "jsonpath" not in sys.modules:
    sys.modules["jsonpath"] = jsonpath_ng

import akshare as ak

logger = logging.getLogger(__name__)

DB_PATH = Path("data/cache.db")
CACHE_TTL_HOURS = 24  # refresh after this many hours


def _with_retry(fn, *args, retries: int = 4, base_delay: float = 2.0, **kwargs):
    """
    Retry an akshare call with exponential backoff.
    eastmoney's anti-bot layer intermittently drops connections
    (RemoteDisconnected) for automated clients — retrying almost
    always succeeds within a couple of attempts.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"{fn.__name__} failed (attempt {attempt+1}/{retries}): {e}. "
                    f"Retrying in {delay:.0f}s …"
                )
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Low-level cache helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _is_fresh(conn: sqlite3.Connection, table: str, key: str = "") -> bool:
    """Return True if table has a row fetched within CACHE_TTL_HOURS."""
    try:
        q = f"SELECT fetched_at FROM {table}_meta WHERE cache_key=?"
        row = conn.execute(q, (key,)).fetchone()
        if row is None:
            return False
        fetched = datetime.fromisoformat(row[0])
        return datetime.utcnow() - fetched < timedelta(hours=CACHE_TTL_HOURS)
    except sqlite3.OperationalError:
        return False


def _touch(conn: sqlite3.Connection, table: str, key: str = "") -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {table}_meta "
        "(cache_key TEXT PRIMARY KEY, fetched_at TEXT)"
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {table}_meta VALUES (?, ?)",
        (key, datetime.utcnow().isoformat()),
    )
    conn.commit()


def _load_df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def _save_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# Step 1: Universe (spot data)
# ---------------------------------------------------------------------------

def fetch_universe(force: bool = False) -> pd.DataFrame:
    """
    Returns A-share universe with market cap, price, change%, volume.
    Columns standardised to English names.
    """
    conn = _get_conn()
    table = "universe"
    if not force and _is_fresh(conn, table):
        df = _load_df(conn, table)
        if not df.empty:
            conn.close()
            return df

    logger.info("Fetching A-share universe from akshare …")
    raw = _with_retry(ak.stock_zh_a_spot_em)

    # Column map (akshare returns Chinese headers)
    col_map = {
        "代码": "ticker",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "pct_change",
        "成交量": "volume",
        "成交额": "turnover",
        "总市值": "mkt_cap",
        "流通市值": "float_mkt_cap",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "60日涨跌幅": "pct_60d",
        "年初至今涨跌幅": "pct_ytd",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    # Numeric coercion
    for col in ["price", "pct_change", "volume", "turnover", "mkt_cap",
                "float_mkt_cap", "pe_ttm", "pb", "pct_60d", "pct_ytd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # mkt_cap in yuan → convert to CNY bn
    if "mkt_cap" in df.columns:
        df["mkt_cap_bn"] = df["mkt_cap"] / 1e8  # akshare gives yuan

    # ST flag: name contains 'ST' or '*ST'
    if "name" in df.columns:
        df["is_st"] = df["name"].str.contains(r"\*?ST", na=False)

    _save_df(conn, df, table)
    _touch(conn, table)
    conn.close()
    logger.info(f"Universe fetched: {len(df)} stocks")
    return df


# ---------------------------------------------------------------------------
# Step 2: ADTV from daily history (sampled)
# ---------------------------------------------------------------------------

def fetch_adtv(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame(ticker, adtv_60d) in CNY.
    Fetches history for tickers not already cached.
    """
    conn = _get_conn()
    table = "adtv"

    if not force and _is_fresh(conn, table, key="adtv"):
        df = _load_df(conn, table)
        if not df.empty and set(tickers).issubset(set(df["ticker"])):
            conn.close()
            return df

    logger.info(f"Fetching 60d history for {len(tickers)} tickers …")
    records = []
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=120)).strftime("%Y%m%d")

    for i, ticker in enumerate(tickers):
        try:
            hist = _with_retry(
                ak.stock_zh_a_hist, retries=2, base_delay=1.5,
                symbol=ticker, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            if hist is not None and not hist.empty:
                # column: '成交额' = turnover in CNY
                tv_col = [c for c in hist.columns if "成交额" in c]
                if tv_col:
                    adtv = hist[tv_col[0]].tail(60).mean()
                else:
                    adtv = np.nan
            else:
                adtv = np.nan
            records.append({"ticker": ticker, "adtv_60d": adtv})
        except Exception as e:
            records.append({"ticker": ticker, "adtv_60d": np.nan})
        if i % 50 == 49:
            logger.info(f"  ADTV progress: {i+1}/{len(tickers)}")
        time.sleep(0.05)  # light rate-limit

    df = pd.DataFrame(records)
    _save_df(conn, df, table)
    _touch(conn, table, key="adtv")
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Step 3: Financials – abstract + indicators per ticker
# ---------------------------------------------------------------------------

def fetch_financials_batch(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """
    Returns one row per ticker with key financial metrics.
    Uses stock_financial_abstract + stock_financial_analysis_indicator.
    """
    conn = _get_conn()
    table = "financials"

    if not force and _is_fresh(conn, table, key="financials"):
        df = _load_df(conn, table)
        if not df.empty:
            conn.close()
            return df

    logger.info(f"Fetching financials for {len(tickers)} tickers …")
    records = []

    for i, ticker in enumerate(tickers):
        rec = {"ticker": ticker}
        try:
            ind = _with_retry(
                ak.stock_financial_analysis_indicator, retries=2, base_delay=1.5,
                symbol=ticker, start_year="2020",
            )
            if ind is not None and not ind.empty:
                # grab most-recent row
                row = ind.iloc[0]
                def safe(col):
                    for c in ind.columns:
                        if col in c:
                            v = row.get(c, np.nan)
                            try:
                                return float(str(v).replace("%","").replace(",",""))
                            except:
                                return np.nan
                    return np.nan

                rec["roe"] = safe("净资产收益率")
                rec["gross_margin"] = safe("销售毛利率")
                rec["net_margin"] = safe("销售净利率")
                rec["asset_turnover"] = safe("总资产周转率")
                rec["current_ratio"] = safe("流动比率")
                rec["debt_to_equity"] = safe("资产负债率")
                rec["rev_growth"] = safe("营业收入增长率")
                rec["eps"] = safe("基本每股收益")

            # grab 8 quarters of gross margin for stability calc
            try:
                ind_all = _with_retry(
                    ak.stock_financial_analysis_indicator, retries=2, base_delay=1.5,
                    symbol=ticker, start_year="2022",
                )
                if ind_all is not None and len(ind_all) >= 4:
                    gm_vals = []
                    for c in ind_all.columns:
                        if "销售毛利率" in c:
                            gm_vals = pd.to_numeric(
                                ind_all[c].astype(str).str.replace("%",""), errors="coerce"
                            ).dropna().tolist()
                            break
                    rec["gross_margin_std"] = np.std(gm_vals) if len(gm_vals) >= 4 else np.nan
                    rec["gross_margin_8q"] = gm_vals[:8] if gm_vals else []
            except:
                pass

        except Exception as e:
            pass

        records.append(rec)
        if i % 50 == 49:
            logger.info(f"  Financials progress: {i+1}/{len(tickers)}")
        time.sleep(0.08)

    df = pd.DataFrame(records)
    _save_df(conn, df, table)
    _touch(conn, table, key="financials")
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Step 4: Cash-flow & accruals data
# ---------------------------------------------------------------------------

def fetch_cashflow_batch(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """
    Fetch operating cash flow, net income, total assets for accruals ratio.
    Uses stock_financial_abstract.
    """
    conn = _get_conn()
    table = "cashflow"

    if not force and _is_fresh(conn, table, key="cashflow"):
        df = _load_df(conn, table)
        if not df.empty:
            conn.close()
            return df

    logger.info(f"Fetching cash-flow data for {len(tickers)} tickers …")
    records = []

    for i, ticker in enumerate(tickers):
        rec = {"ticker": ticker}
        try:
            ab = _with_retry(ak.stock_financial_abstract, retries=2, base_delay=1.5, symbol=ticker)
            if ab is not None and not ab.empty:
                def get_item(df, *keys):
                    for key in keys:
                        rows = df[df.iloc[:, 0].astype(str).str.contains(key, na=False)]
                        if not rows.empty:
                            # columns after first are year values
                            vals = rows.iloc[0, 1:].apply(
                                lambda x: float(str(x).replace(",","").replace("--",""))
                                if str(x) not in ["--", "nan", "None", ""] else np.nan
                            )
                            return vals.dropna().tolist()
                    return []

                ocf = get_item(ab, "经营活动产生的现金流量净额", "经营活动现金流量净额")
                ni  = get_item(ab, "净利润", "归属于母公司所有者的净利润")
                ta  = get_item(ab, "资产总计", "总资产")
                rec_gr = get_item(ab, "应收账款", "应收票据及应收账款")
                rev   = get_item(ab, "营业收入", "营业总收入")
                div   = get_item(ab, "每股股利", "股息", "分红")

                rec["ocf_list"] = ocf[:4]
                rec["ni_list"]  = ni[:4]
                rec["ta_list"]  = ta[:4]
                rec["rec_list"] = rec_gr[:3]
                rec["rev_list"] = rev[:3]
                rec["div_list"] = div[:4]

                # Derived scalars
                if ocf and ni and ta and ta[0]:
                    rec["accruals_ratio"] = (ni[0] - ocf[0]) / ta[0] if ta[0] else np.nan
                if ocf and ni:
                    valid_pairs = [(o, n) for o, n in zip(ocf[:4], ni[:4])
                                   if n and n != 0]
                    if valid_pairs:
                        rec["cash_conversion"] = np.mean([o/n for o,n in valid_pairs])
                    # flag if <0.7 for 2 consecutive years
                    cc_flags = [o/n < 0.7 for o,n in valid_pairs if n != 0]
                    rec["cc_red_flag"] = sum(cc_flags[:2]) == 2 if len(cc_flags) >= 2 else False

                # Receivables growth vs revenue growth
                if len(rec_gr) >= 2 and len(rev) >= 2 and rev[1] and rec_gr[1]:
                    rev_g = (rev[0] - rev[1]) / abs(rev[1])
                    rec_g = (rec_gr[0] - rec_gr[1]) / abs(rec_gr[1])
                    rec["rec_growth"] = rec_g
                    rec["rev_growth_cf"] = rev_g
                    rec["rec_vs_rev_flag"] = rec_g > 1.5 * rev_g if rev_g > 0 else False

                # Dividend yield / payout (3-year growth)
                if div and len(div) >= 2:
                    rec["div_3y_growth"] = (div[0] - div[-1]) >= 0 if div[-1] else False

                # FCF proxy: OCF positive in 3 of last 3 years
                if ocf:
                    rec["fcf_positive_3y"] = sum(1 for x in ocf[:3] if x and x > 0) >= 3

        except Exception as e:
            pass

        records.append(rec)
        if i % 50 == 49:
            logger.info(f"  Cashflow progress: {i+1}/{len(tickers)}")
        time.sleep(0.08)

    df = pd.DataFrame(records)
    _save_df(conn, df, table)
    _touch(conn, table, key="cashflow")
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Step 5: Valuation time-series (PE/PB percentile)
# ---------------------------------------------------------------------------

def fetch_valuation_percentile(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """
    Returns PE/PB current vs 5-year percentile.
    """
    conn = _get_conn()
    table = "valuation"

    if not force and _is_fresh(conn, table, key="valuation"):
        df = _load_df(conn, table)
        if not df.empty:
            conn.close()
            return df

    logger.info(f"Fetching valuation history for {len(tickers)} tickers …")
    records = []

    for i, ticker in enumerate(tickers):
        rec = {"ticker": ticker}
        try:
            val = _with_retry(ak.stock_a_indicator_lg, retries=2, base_delay=1.5, symbol=ticker)
            if val is not None and not val.empty:
                pe_col = [c for c in val.columns if "pe" in c.lower() or "市盈" in c]
                pb_col = [c for c in val.columns if "pb" in c.lower() or "市净" in c]
                if pe_col:
                    pe_series = pd.to_numeric(val[pe_col[0]], errors="coerce").dropna()
                    if len(pe_series) > 10:
                        rec["pe_current"] = pe_series.iloc[-1]
                        rec["pe_pct"] = (pe_series <= pe_series.iloc[-1]).mean() * 100
                        rec["pe_5y_low"] = pe_series.quantile(0.05)
                        rec["pe_5y_high"] = pe_series.quantile(0.95)
                if pb_col:
                    pb_series = pd.to_numeric(val[pb_col[0]], errors="coerce").dropna()
                    if len(pb_series) > 10:
                        rec["pb_current"] = pb_series.iloc[-1]
                        rec["pb_pct"] = (pb_series <= pb_series.iloc[-1]).mean() * 100
        except Exception:
            pass

        records.append(rec)
        if i % 50 == 49:
            logger.info(f"  Valuation progress: {i+1}/{len(tickers)}")
        time.sleep(0.08)

    df = pd.DataFrame(records)
    _save_df(conn, df, table)
    _touch(conn, table, key="valuation")
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Step 6: Industry membership (for localization theme)
# ---------------------------------------------------------------------------

LOCALIZATION_BOARDS = [
    "半导体", "集成电路", "芯片", "电子元件", "工业自动化",
    "机器人", "锂电池", "电池材料", "新材料", "光伏设备",
    "储能", "精密制造", "数控机床", "航空装备",
]

def fetch_industry_tags(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """
    Tag tickers with localization/self-sufficiency flag via industry board.
    """
    conn = _get_conn()
    table = "industry"

    if not force and _is_fresh(conn, table, key="industry"):
        df = _load_df(conn, table)
        if not df.empty:
            conn.close()
            return df

    logger.info("Fetching industry board list …")
    records = []

    try:
        boards = _with_retry(ak.stock_board_industry_name_em, retries=3, base_delay=2.0)
        board_names = boards["板块名称"].tolist() if "板块名称" in boards.columns else []
    except Exception:
        board_names = []

    # Build a ticker→board mapping for localization boards
    ticker_board = {}
    localization_boards_found = [b for b in board_names
                                   if any(kw in b for kw in LOCALIZATION_BOARDS)]

    for board in localization_boards_found[:30]:  # cap to avoid excess calls
        try:
            members = _with_retry(
                ak.stock_board_industry_cons_em, retries=2, base_delay=1.5, symbol=board
            )
            if members is not None and not members.empty:
                code_col = [c for c in members.columns if "代码" in c]
                if code_col:
                    for t in members[code_col[0]].tolist():
                        ticker_board.setdefault(str(t), []).append(board)
            time.sleep(0.1)
        except Exception:
            continue

    for ticker in tickers:
        boards_for = ticker_board.get(ticker, [])
        records.append({
            "ticker": ticker,
            "industry_boards": ";".join(boards_for),
            "is_localization": len(boards_for) > 0,
        })

    df = pd.DataFrame(records)
    _save_df(conn, df, table)
    _touch(conn, table, key="industry")
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Step 7: Individual stock info (for dividend yield from spot)
# ---------------------------------------------------------------------------

def fetch_dividend_yield(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dividend yield from spot data if available, else NaN.
    akshare stock_zh_a_spot_em sometimes includes 股息率 column.
    """
    div_col = [c for c in universe_df.columns if "股息" in str(c) or "dividend" in str(c).lower()]
    if div_col:
        return universe_df[["ticker", div_col[0]]].rename(columns={div_col[0]: "div_yield"})
    # Try stock_a_indicator_lg for current yield on each ticker – expensive,
    # so we'll use the PE/PB fetch path and pull it there.
    return pd.DataFrame({"ticker": universe_df["ticker"], "div_yield": np.nan})


# ---------------------------------------------------------------------------
# Convenience: full refresh
# ---------------------------------------------------------------------------

def cache_status() -> dict:
    """Return dict of table → (row_count, fetched_at) for display."""
    conn = _get_conn()
    status = {}
    for table in ["universe", "adtv", "financials", "cashflow", "valuation", "industry"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            meta = conn.execute(
                f"SELECT fetched_at FROM {table}_meta WHERE cache_key=?", (
                    "adtv" if table == "adtv" else
                    "financials" if table == "financials" else
                    "cashflow" if table == "cashflow" else
                    "valuation" if table == "valuation" else
                    "industry" if table == "industry" else "",
                )
            ).fetchone()
            fetched = meta[0] if meta else "never"
        except:
            count, fetched = 0, "never"
        status[table] = {"rows": count, "fetched_at": fetched}
    conn.close()
    return status
