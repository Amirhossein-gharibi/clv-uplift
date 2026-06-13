# src/clv_uplift/data/loader.py
"""
Phase 1.1 - Transaction loading and cleaning.

Provides clean_transactions(): a five-step cleaning pipeline applied in a fixed,
auditable order, followed by a hard temporal-validity check against the configured
snapshot date. Before/after row counts are printed at every step, per the project's
coding conventions.

Column reconciliation: the raw 2010-2011 sheet ships columns named 'Invoice',
'Price', and 'Customer ID', whereas the handoff's cleaning spec references
'InvoiceNo', 'UnitPrice', and 'CustomerID'. The first action in cleaning renames the
raw columns to these canonical names so the rest of the pipeline - and every
downstream phase - speaks one consistent vocabulary.

cancel_count: a per-customer count of cancelled invoices is extracted from the raw
data BEFORE cancellation rows are dropped (step 1 deletes them), then merged back onto
the cleaned purchase-only transactions. build_rfm() uses it to derive cancel_rate.
The five cleaning steps and their row counts are unaffected by this - cancel_count is
an additional column that rides along.

Run standalone (loads the raw sheet, then cleans, printing the full log):
    python -m clv_uplift.data.loader
"""
from __future__ import annotations

import pandas as pd

from clv_uplift.config import SNAPSHOT_DATE_STR
from clv_uplift.data.audit import load_raw

# Raw -> canonical column names. Columns not listed here are kept as-is.
COLUMN_RENAME_MAP = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}

# A legitimate product StockCode begins with a 5-digit code (e.g. '85123A', '47566B').
# Non-product service codes (POST, DOT, D, C2, CRUK, AMAZONFEE, BANK CHARGES, M, S,
# gift vouchers, ...) do not begin with five digits and are removed in step 3.
PRODUCT_STOCKCODE_PATTERN = r"^\d{5}"

# Columns that must exist (under canonical names) for the pipeline to run.
REQUIRED_COLUMNS = {
    "InvoiceNo", "CustomerID", "StockCode", "UnitPrice", "Quantity", "InvoiceDate",
}


def _rule(char: str = "=", width: int = 70) -> str:
    return char * width


def _log_step(label: str, before: int, after: int) -> None:
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    print(f"[{label}]")
    print(f"    before : {before:,}")
    print(f"    after  : {after:,}")
    print(f"    removed: {removed:,} ({pct:.2f}%)")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the five-step cleaning pipeline in fixed order, attach cancel_count, then run
    the temporal check.

    Steps (order is significant - each feeds the next):
      1. Remove cancellations            (InvoiceNo starting with 'C')
      2. Remove rows with missing CustomerID
      3. Remove non-product StockCodes   (StockCode not beginning with a 5-digit code)
      4. Remove rows with non-positive UnitPrice  (<= 0)
      5. Remove rows with negative Quantity       (< 0; zero is retained)

    Returns the cleaned DataFrame with canonical column names, a cancel_count column,
    and a reset index.
    """
    print(_rule())
    print("CLEANING TRANSACTIONS (clean_transactions)")
    print(_rule())

    df = df.rename(columns=COLUMN_RENAME_MAP)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            f"Expected columns missing after rename: {sorted(missing)}. "
            f"Columns present: {list(df.columns)}. "
            f"Reconcile COLUMN_RENAME_MAP in loader.py before proceeding."
        )

    n_start = len(df)
    print(f"Starting rows: {n_start:,}\n")

    # --- Extract cancel_count BEFORE dropping cancellations (step 1 deletes them) ---
    # A cancellation is an InvoiceNo starting with 'C'. Count distinct cancelled
    # invoices per identified customer; cancellations with a missing CustomerID are
    # dropped by the groupby (cancellations can only be attributed to known customers).
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    cancel_counts = (
        df.loc[is_cancellation]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .rename("cancel_count")
        .reset_index()
    )
    print(f"cancel_count extracted: {len(cancel_counts):,} customers had cancellations "
          f"({int(cancel_counts['cancel_count'].sum()):,} cancelled invoices total)\n")

    # Step 1 - remove cancellations (InvoiceNo starting with 'C')
    before = len(df)
    df = df.loc[~is_cancellation]
    _log_step("1/5 remove cancellations (InvoiceNo starts with 'C')", before, len(df))

    # Step 2 - remove rows with missing CustomerID
    before = len(df)
    df = df.loc[df["CustomerID"].notna()]
    _log_step("2/5 remove missing CustomerID", before, len(df))

    # Step 3 - remove non-product StockCodes (keep only StockCodes matching '^\d{5}')
    before = len(df)
    is_product = df["StockCode"].astype(str).str.match(PRODUCT_STOCKCODE_PATTERN, na=False)
    df = df.loc[is_product]
    _log_step("3/5 remove non-product StockCodes (keep '^\\d{5}')", before, len(df))

    # Step 4 - remove non-positive UnitPrice (<= 0)
    before = len(df)
    df = df.loc[df["UnitPrice"] > 0]
    _log_step("4/5 remove non-positive UnitPrice (<= 0)", before, len(df))

    # Step 5 - remove negative Quantity (< 0; zero retained per literal spec)
    before = len(df)
    df = df.loc[df["Quantity"] >= 0]
    _log_step("5/5 remove negative Quantity (< 0)", before, len(df))

    n_end = len(df)
    net_removed = n_start - n_end
    print(f"\nNet: {n_start:,} -> {n_end:,} "
          f"({net_removed:,} removed, {net_removed / n_start * 100:.2f}%)")

    # --- Merge cancel_count back onto the cleaned (purchase-only) transactions -----
    df = df.merge(cancel_counts, on="CustomerID", how="left")
    df["cancel_count"] = df["cancel_count"].fillna(0).astype(int)
    n_with_cancels = int((df.groupby("CustomerID")["cancel_count"].first() > 0).sum())
    n_customers = df["CustomerID"].nunique()
    print(f"cancel_count merged: {n_with_cancels:,}/{n_customers:,} retained customers "
          f"have >= 1 cancellation")

    # --- Temporal validity check (hard causal-validity rule) ----------------------
    print("\n" + _rule("-"))
    print("TEMPORAL CHECK (max InvoiceDate <= snapshot)")
    print(_rule("-"))
    snapshot = pd.Timestamp(SNAPSHOT_DATE_STR)
    invoice_dates = pd.to_datetime(df["InvoiceDate"])
    max_date = invoice_dates.max()
    print(f"max(InvoiceDate): {max_date}")
    print(f"snapshot date  : {snapshot} (from config)")
    if pd.notna(max_date) and max_date > snapshot:
        n_future = int((invoice_dates > snapshot).sum())
        print(f"WARNING: {n_future:,} transactions occur AFTER the snapshot date. "
              f"This violates the temporal-validity rule and would corrupt RFM recency. "
              f"Investigate before Phase 1.2.")
    else:
        print("OK: all cleaned transactions fall on or before the snapshot date.")

    print("\n" + _rule())
    print("CLEANING COMPLETE")
    print(_rule())
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    clean_transactions(raw)