# research_splits.py
# Train/Val/Test (with embargo) + optional purged k-folds for EGARCH BTC work.

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# ---- Defaults for YOUR project ----
DEFAULT_DATA = "C:/Users/cpase/OneDrive/Υπολογιστής/Projects/EGARCH BTC USDT/BTCUSDTmergeddataset.csv"
DEFAULT_DATE_COL = "timestamp"
DEFAULT_TRAIN_END = "2021-12-31"
DEFAULT_VAL_END   = "2023-12-31"
DEFAULT_EMBARGO   = 5

def _infer_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["timestamp", "date", "datetime", "time"]:
        if c in df.columns:
            return c
    return None

def load_series(path: Path, date_col: Optional[str] = None, resample: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] Data file not found: {path}")
    peek = pd.read_csv(path, nrows=5)
    if date_col is None:
        date_col = _infer_date_col(peek)
    if not date_col or date_col not in peek.columns:
        sys.exit(f"[error] Date column '{date_col}' not found in file.")

    df = None
    for dayfirst in (False, True):
        try:
            tmp = pd.read_csv(path, parse_dates=[date_col], dayfirst=dayfirst, thousands=",")
            tmp = tmp.set_index(date_col).sort_index()
            if isinstance(tmp.index, pd.DatetimeIndex):
                df = tmp
                break
        except Exception:
            continue
    if df is None or df.empty:
        sys.exit("[error] Could not parse dates or dataframe is empty. Consider --date-col and checking the CSV.")

    # Drop duplicate timestamps, keep first
    df = df[~df.index.duplicated(keep="first")]

    # Optional resample (e.g., W-MON), last observation per period (close-to-close logic)
    if resample:
        how = "last"
        df = df.resample(resample).agg(how).dropna(how="all")

    if df.empty:
        sys.exit("[error] Data empty after resampling. Check --resample.")
    return df

def _pos_after(ts_index: pd.DatetimeIndex, cutoff: pd.Timestamp, side="right") -> int:
    return ts_index.searchsorted(cutoff, side=side)

def make_splits_by_dates(df: pd.DataFrame, train_end: str, val_end: str, embargo: int = 0):
    df = df.sort_index()
    idx = df.index
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts   = pd.to_datetime(val_end)
    if not (idx[0] < train_end_ts < val_end_ts < idx[-1]):
        sys.exit("[error] Require: start < train_end < val_end < end (check your dates).")

    train_end_pos = _pos_after(idx, train_end_ts, side="right")
    val_end_pos   = _pos_after(idx, val_end_ts,   side="right")

    start_val_pos  = min(train_end_pos + max(0, embargo), len(df))
    start_test_pos = min(val_end_pos   + max(0, embargo), len(df))

    train = df.iloc[:train_end_pos].copy()
    val   = df.iloc[start_val_pos:val_end_pos].copy()
    test  = df.iloc[start_test_pos:].copy()

    _assert_no_overlap(train, val, test)
    return train, val, test

def make_splits_by_props(df: pd.DataFrame, props: Tuple[float, float, float], embargo: int = 0):
    """props = (train, val, test) fractions that sum to 1."""
    df = df.sort_index()
    T = len(df)
    if T < 100:
        sys.exit("[error] Not enough rows for proportion splits (need >=100).")
    a, b, c = props
    if not np.isclose(a + b + c, 1.0):
        sys.exit("[error] Proportions must sum to 1 (e.g., 0.7 0.15 0.15).")

    p1 = int(round(T * a))
    p2 = int(round(T * (a + b)))

    start_val = min(p1 + embargo, T)
    start_test = min(p2 + embargo, T)

    train = df.iloc[:p1].copy()
    val   = df.iloc[start_val:p2].copy()
    test  = df.iloc[start_test:].copy()

    _assert_no_overlap(train, val, test)
    return train, val, test

def _assert_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, min_len: int = 50):
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if len(part) < min_len:
            print(f"[warn] {name} has only {len(part)} rows (<{min_len}).", file=sys.stderr)
    def _last(d): return None if d.empty else d.index[-1]
    def _first(d): return None if d.empty else d.index[0]
    if not train.empty and not val.empty and _last(train) >= _first(val):
        sys.exit("[error] Overlap: train and validation after embargo.")
    if not val.empty and not test.empty and _last(val) >= _first(test):
        sys.exit("[error] Overlap: validation and test after embargo.")

def save_split_csvs(train, val, test, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    train.to_csv(outdir / "train.csv")
    val.to_csv(outdir / "validation.csv")
    test.to_csv(outdir / "test.csv")

def _safe_dates(s: pd.DataFrame) -> Tuple[str, str]:
    if s.empty:
        return ("N/A", "N/A")
    return (str(s.index[0].date()), str(s.index[-1].date()))

def render_markdown(train, val, test, embargo, freq_label: str):
    t0, t1 = _safe_dates(train); v0, v1 = _safe_dates(val); o0, o1 = _safe_dates(test)
    def _span(s): 
        if s.empty: return "0 (0.0%)"
        total = len(train) + len(val) + len(test)
        return f"{len(s)} ({len(s)/max(1,total):.1%})"
    md = f"""
### Train / Validation / Test Protocol

- **Frequency:** {freq_label}
- **Train:** {t0} → {t1}  ({_span(train)})
- **Validation:** {v0} → {v1}  ({_span(val)})
- **Test (Out-of-Sample):** {o0} → {o1}  ({_span(test)})
- **Embargo:** {embargo} bar(s) at each boundary to reduce information leakage due to serial dependence.

**Why these periods?**
- **2019–2020 (Train):** mixed/sideways markets with a gradual uptrend give EGARCH exposure to low–mid volatility regimes.
- **2021–2022 (Validation):** extreme bull (2021) followed by a deep bear (2022) stress-test the hyperparameters (take-profit %, stop k·σ, refit cadence) across opposite regimes.
- **2023–present (Test):** recovery and new ATHs with more stepwise advances. We freeze hyperparameters after validation and evaluate **once** on this unseen period.

**Protocol details**
- Hyperparameters are chosen **only** on the validation set (robustness plateaus preferred over single maxima).
- Model parameters (EGARCH coefficients) are **re-estimated walk-forward** using only past data at each step.
- The test set is untouched until the end and is evaluated in a single pass.
""".strip()
    return md

# ---------------- Purged K-Folds (time series) ----------------
def purged_kfold_indices(T: int, n_splits: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) for each fold with an embargo around the test block.
    Each fold's test block is a contiguous segment; training excludes the embargo region.
    """
    if n_splits < 2 or T < n_splits:
        raise ValueError("n_splits must be >=2 and <= T")
    fold_sizes = np.full(n_splits, T // n_splits, dtype=int)
    fold_sizes[: T % n_splits] += 1
    starts = np.cumsum(np.r_[0, fold_sizes[:-1]])
    out = []
    for k, (st, sz) in enumerate(zip(starts, fold_sizes)):
        test_idx = np.arange(st, st + sz)
        left = max(0, st - embargo)
        right = min(T, st + sz + embargo)
        train_idx = np.r_[np.arange(0, left), np.arange(right, T)]
        out.append((train_idx, test_idx))
    return out

def save_kfolds(df: pd.DataFrame, n_splits: int, embargo: int, outdir: Path):
    T = len(df)
    splits = purged_kfold_indices(T, n_splits, embargo)
    base = outdir / f"kfolds_{n_splits}"
    base.mkdir(parents=True, exist_ok=True)
    for i, (tr, te) in enumerate(splits, 1):
        fold_dir = base / f"fold_{i:02d}"
        fold_dir.mkdir(exist_ok=True)
        df.iloc[tr].to_csv(fold_dir / "train.csv")
        df.iloc[te].to_csv(fold_dir / "test.csv")

def main():
    p = argparse.ArgumentParser(description="Create train/val/test splits with embargo, and optional purged k-folds.")
    p.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV.")
    p.add_argument("--date-col", type=str, default=DEFAULT_DATE_COL, help="Column to use as datetime index.")
    p.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END, help="Calendar cut for training set end.")
    p.add_argument("--val-end", type=str, default=DEFAULT_VAL_END, help="Calendar cut for validation set end.")
    p.add_argument("--embargo", type=int, default=DEFAULT_EMBARGO, help="Bars to embargo around boundaries.")
    p.add_argument("--outdir", type=str, default="results/splits")
    p.add_argument("--readme-out", type=str, default="results/README_split_snippet.md")
    p.add_argument("--resample", type=str, default=None, help='e.g., "W-MON"; if omitted, use raw frequency.')
    p.add_argument("--props", type=float, nargs=3, default=None,
                   help="Alternative to dates: three proportions that sum to 1.0 (e.g., 0.7 0.15 0.15)")
    p.add_argument("--kfolds", type=int, default=0, help="If >0, also emit purged k-folds with this many splits.")
    args = p.parse_args()

    df = load_series(Path(args.data), date_col=args.date_col, resample=args.resample)
    freq_label = args.resample or "raw"

    if args.props is not None:
        train, val, test = make_splits_by_props(df, tuple(args.props), embargo=args.embargo)
    else:
        train, val, test = make_splits_by_dates(df, args.train_end, args.val_end, embargo=args.embargo)

    save_split_csvs(train, val, test, Path(args.outdir))
    md = render_markdown(train, val, test, args.embargo, freq_label=freq_label)

    readme_path = Path(args.readme_out)
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(md, encoding="utf-8")

    # Integrity report
    def _desc(s: pd.DataFrame, name: str):
        if s.empty:
            return f"{name}: 0 rows"
        return f"{name}: {len(s)} rows | {s.index[0].date()} → {s.index[-1].date()}"
    print("\n--- README SNIPPET START ---\n")
    print(md)
    print("\n--- README SNIPPET END ---\n")
    print(_desc(train, "Train"))
    print(_desc(val,   "Validation"))
    print(_desc(test,  "Test"))
    # Check embargo distances if all non-empty
    if not train.empty and not val.empty:
        gap_tv = (val.index[0] - train.index[-1]).days
        print(f"Gap (train→val) = {gap_tv} day(s) [embargo={args.embargo} bars]")
    if not val.empty and not test.empty:
        gap_vt = (test.index[0] - val.index[-1]).days
        print(f"Gap (val→test)  = {gap_vt} day(s) [embargo={args.embargo} bars]")

    print(f"\nSaved CSV splits to: {args.outdir}")
    print(f"Saved README snippet to: {args.readme_out}")

    # Optional k-folds
    if args.kfolds and args.kfolds > 1:
        save_kfolds(df, n_splits=args.kfolds, embargo=args.embargo, outdir=Path(args.outdir))
        print(f"Saved purged {args.kfolds}-folds under {args.outdir}/kfolds_{args.kfolds}")

if __name__ == "__main__":
    main()
