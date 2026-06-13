# src/clv_uplift/data/audit.py
"""
Phase 1.0 - Data Quality Audit.

Purely descriptive. Performs NO transformations. Loads the raw UCI Online Retail II
workbook (2010-2011 trading-year sheet only, matching SNAPSHOT_DATE_STR) and prints a
summary for interpretation before any cleaning happens in Phase 1.1.

Logged per the handoff: row count, dtypes, missing-value counts by column, and
CustomerID cardinality. Columns above 30% missingness are flagged for documented
removal. Treatment-variable distribution is intentionally deferred to a post-Phase-1.3
audit, since treatment does not exist in the raw file and computing it here would be a
transformation this phase forbids.

Run:  python -m clv_uplift.data.audit
"""
from __future__ import annotations

import pandas as pd

from clv_uplift.config import DATA_DIR, UCI_FILENAME, SNAPSHOT_DATE_STR

# Columns above this missingness fraction are flagged for documented removal.
MISSINGNESS_FLAG_THRESHOLD = 0.30

# Distinguishing token for the 2010-2011 sheet. The 2009-2010 sheet also contains
# "2010", so we key off "2011", which only the target sheet contains.
TARGET_SHEET_TOKEN = "2011"

# Schema the Phase 1.1 loader currently expects (the older "Online Retail" naming).
# Reported here only to surface naming differences for reconciliation in Phase 1.1.
LOADER_EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def _resolve_sheet_name(xls: pd.ExcelFile) -> str:
    """Select the 2010-2011 sheet by token match. Raise (do not guess) on ambiguity."""
    sheets = list(xls.sheet_names)
    print("Available sheets in workbook:")
    for s in sheets:
        print(f"  - {s!r}")

    candidates = [s for s in sheets if TARGET_SHEET_TOKEN in str(s)]
    if len(candidates) == 1:
        chosen = candidates[0]
        print(f"\nSelected sheet (contains {TARGET_SHEET_TOKEN!r}): {chosen!r}")
        return chosen

    raise ValueError(
        f"Could not resolve a single 2010-2011 sheet by token {TARGET_SHEET_TOKEN!r}. "
        f"Matches found: {candidates}. Available sheets: {sheets}. "
        f"Tell the implementation chat the exact sheet name to use."
    )


def load_raw(sheet_name: str | None = None) -> pd.DataFrame:
    """
    Load the raw 2010-2011 sheet. No cleaning, no transformation.

    If sheet_name is None, the sheet is auto-resolved by token match.
    """
    path = DATA_DIR / UCI_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download online_retail_II.xlsx from "
            f"https://archive.ics.uci.edu/dataset/502/online+retail+ii and place it in "
            f"{DATA_DIR}."
        )

    print(_rule())
    print("LOADING RAW WORKBOOK (no transformations)")
    print(_rule())
    print(f"Path: {path}")

    with pd.ExcelFile(path) as xls:
        resolved = sheet_name or _resolve_sheet_name(xls)
        df = pd.read_excel(xls, sheet_name=resolved)

    print(f"Loaded sheet {resolved!r}: {len(df):,} rows x {df.shape[1]} columns")
    return df


def _detect_column(df: pd.DataFrame, *aliases: str) -> str | None:
    """Find a column by case/space-insensitive match against any alias."""
    norm = {c.lower().replace(" ", ""): c for c in df.columns}
    for alias in aliases:
        key = alias.lower().replace(" ", "")
        if key in norm:
            return norm[key]
    return None


def audit(sheet_name: str | None = None) -> pd.DataFrame:
    """Run the descriptive audit, print the summary, and return the raw DataFrame."""
    df = load_raw(sheet_name=sheet_name)
    n_rows = len(df)

    # --- Columns present vs. what the Phase 1.1 loader expects -----------------
    print("\n" + _rule())
    print("COLUMN SCHEMA")
    print(_rule())
    print("Columns present in this sheet:")
    for c in df.columns:
        print(f"  - {c!r}")

    missing_expected = [c for c in LOADER_EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected:
        print("\nNOTE: Phase 1.1 loader expects these names that are ABSENT here:")
        for c in missing_expected:
            print(f"  - {c}")
        print("These will be reconciled when loader.py is written in Phase 1.1.")
    else:
        print("\nAll Phase 1.1 loader column names are present as-is.")

    # --- Per-column dtype + missingness ---------------------------------------
    print("\n" + _rule())
    print("DTYPES AND MISSING-VALUE COUNTS")
    print(_rule())
    missing_count = df.isna().sum()
    missing_pct = (missing_count / n_rows * 100).round(2)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": missing_count,
        "missing_pct": missing_pct,
    })
    print(summary.to_string())
    print(f"\nTotal rows: {n_rows:,}")

    # --- 30% missingness flag --------------------------------------------------
    flagged = summary.index[summary["missing_pct"] > MISSINGNESS_FLAG_THRESHOLD * 100].tolist()
    print("\n" + _rule())
    print(f"MISSINGNESS FLAG (threshold > {MISSINGNESS_FLAG_THRESHOLD:.0%})")
    print(_rule())
    if flagged:
        print("Columns flagged for documented removal:")
        for c in flagged:
            print(f"  - {c}: {summary.loc[c, 'missing_pct']:.2f}% missing")
    else:
        print("No columns exceed the 30% missingness threshold.")

    # --- CustomerID cardinality ------------------------------------------------
    print("\n" + _rule())
    print("CUSTOMER ID CARDINALITY")
    print(_rule())
    cust_col = _detect_column(df, "CustomerID", "Customer ID")
    if cust_col is None:
        print("No CustomerID-like column detected.")
    else:
        n_unique = df[cust_col].nunique(dropna=True)
        n_missing = int(df[cust_col].isna().sum())
        print(f"Column                : {cust_col!r}")
        print(f"Unique customers      : {n_unique:,}")
        print(f"Rows with missing ID  : {n_missing:,} ({n_missing / n_rows * 100:.2f}%)")

    # --- Date range (sheet-selection validation; not in handoff logging list) --
    print("\n" + _rule())
    print("DATE RANGE (sheet-selection validation)")
    print(_rule())
    date_col = _detect_column(df, "InvoiceDate")
    if date_col is None:
        print("No InvoiceDate-like column detected.")
    else:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        dmin, dmax = dates.min(), dates.max()
        print(f"Column        : {date_col!r}")
        print(f"Earliest date : {dmin}")
        print(f"Latest date   : {dmax}")
        print(f"Snapshot date : {SNAPSHOT_DATE_STR} (from config)")
        snapshot = pd.Timestamp(SNAPSHOT_DATE_STR)
        if pd.notna(dmax) and dmax > snapshot:
            print("WARNING: latest InvoiceDate exceeds the snapshot date. "
                  "This will be enforced as a hard temporal check in Phase 1.1.")
        else:
            print("OK: all dates fall on or before the snapshot date.")

    print("\n" + _rule())
    print("AUDIT COMPLETE - no transformations performed. Paste this output back.")
    print(_rule())
    return df


if __name__ == "__main__":
    audit()