# tools/csv_editor.py
# -*- coding: utf-8 -*-
"""
CSV editor utilities for a product catalog.

Key features:
- Load a CSV (UTF-8) with columns like: id, title, category, price,
  rating_rate, rating_count, description, image.
- Apply filters: include/exclude categories, price range, minimum rating.
- Select or drop columns.
- Deduplicate by columns.
- Sort by predefined policies: alphabetical, price, rating, category.
- Save to a sanitized file name (no path traversal), as CSV (default) or XLSX.

Designed to be called directly or wrapped as a LangChain `@tool` in the main app.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import re
import pandas as pd

__all__ = ["edit_products_csv"]


# --------------------------- Filename safety ---------------------------------

_SAFE_NAME_RE = re.compile(
    r"^[\w\-. ]+$"
)  # letters, digits, underscore, dash, dot, space


def _safe_filename(name: str, fmt: str = "csv") -> str:
    """Return a sanitized base filename with the right extension, no paths allowed.

    - Rejects path separators, drive prefixes, and parent traversals.
    - Restricts characters to letters, digits, underscore, dash, dot, and space.
    - Forces extension based on `fmt`.

    Examples:
        _safe_filename("catalog", "csv")    -> "catalog.csv"
        _safe_filename("my export", "xlsx") -> "my export.xlsx"
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`output_name` must be a non-empty string.")

    raw = name.strip()

    # Disallow any path-like patterns
    if os.path.basename(raw) != raw:
        raise ValueError("`output_name` must not contain path separators.")
    if raw.startswith((".", "~")) or ".." in raw:
        raise ValueError(
            "`output_name` must not start with '.' or '~' or contain '..'."
        )
    if "/" in raw or "\\" in raw or ":" in raw:
        raise ValueError(
            "`output_name` must not contain '/', '\\\\', or ':' characters."
        )

    # Restrict character set
    if not _SAFE_NAME_RE.match(raw):
        raise ValueError("`output_name` contains unsupported characters.")

    fmt = (fmt or "csv").lower()
    if fmt not in {"csv", "xlsx"}:
        raise ValueError("`file_format` must be 'csv' or 'xlsx'.")

    ext = ".csv" if fmt == "csv" else ".xlsx"
    base = raw
    if base.lower().endswith((".csv", ".xlsx")):
        base = base[: base.lower().rfind(".")]  # drop existing extension

    return f"{base}{ext}"


# --------------------------- Data normalization ------------------------------


def _normalize_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure rating columns exist with standard names: rating_rate, rating_count."""
    if "rating_rate" not in df.columns:
        if "rating.rate" in df.columns:
            df = df.rename(columns={"rating.rate": "rating_rate"})
        else:
            df["rating_rate"] = pd.NA
    if "rating_count" not in df.columns:
        if "rating.count" in df.columns:
            df = df.rename(columns={"rating.count": "rating_count"})
        else:
            df["rating_count"] = pd.NA
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic normalization for expected product schema."""
    df = _normalize_rating_columns(df)
    return df


# --------------------------- Filters & Ops -----------------------------------


def _apply_filters(
    df: pd.DataFrame,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter rows according to provided rules. Returns filtered df and applied ops."""
    applied: List[str] = []
    out = df.copy()

    # Category normalization (case-insensitive)
    if "category" in out.columns:
        cat_series = out["category"].astype(str).str.strip()
    else:
        cat_series = pd.Series([""] * len(out), index=out.index)

    if include_categories:
        include_set = {c.strip().lower() for c in include_categories if str(c).strip()}
        mask = cat_series.str.lower().isin(include_set)
        out = out[mask]
        applied.append(f"include_categories={sorted(include_set)}")

    if exclude_categories:
        exclude_set = {c.strip().lower() for c in exclude_categories if str(c).strip()}
        mask = ~cat_series.str.lower().isin(exclude_set)
        out = out[mask]
        applied.append(f"exclude_categories={sorted(exclude_set)}")

    if min_price is not None and "price" in out.columns:
        out = out[pd.to_numeric(out["price"], errors="coerce") >= float(min_price)]
        applied.append(f"min_price={min_price}")

    if max_price is not None and "price" in out.columns:
        out = out[pd.to_numeric(out["price"], errors="coerce") <= float(max_price)]
        applied.append(f"max_price={max_price}")

    if min_rating is not None and "rating_rate" in out.columns:
        out = out[
            pd.to_numeric(out["rating_rate"], errors="coerce") >= float(min_rating)
        ]
        applied.append(f"min_rating={min_rating}")

    return out, applied


def _apply_column_selection(
    df: pd.DataFrame,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select or drop columns safely. Returns df and applied ops."""
    applied: List[str] = []
    out = df.copy()

    if keep_columns:
        # Keep intersection only (avoid KeyError)
        keep = [c for c in keep_columns if c in out.columns]
        out = (
            out[keep] if keep else out.iloc[:, 0:0]
        )  # empty df with no columns if none match
        applied.append(f"keep_columns={keep}")

    if drop_columns:
        drop = [c for c in drop_columns if c in out.columns]
        out = out.drop(columns=drop)
        applied.append(f"drop_columns={drop}")

    return out, applied


def _apply_deduplication(
    df: pd.DataFrame, dedupe_on: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop duplicate rows based on subset of columns or entire row if None."""
    applied: List[str] = []
    out = df.copy()

    if dedupe_on:
        subset = [c for c in dedupe_on if c in out.columns]
        out = out.drop_duplicates(subset=subset, keep="first")
        applied.append(f"dedupe_on={subset}")
    else:
        out = out.drop_duplicates(keep="first")
        applied.append("dedupe_on=ALL_COLUMNS")

    return out, applied


def _apply_sort(
    df: pd.DataFrame, sort_order: Optional[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Sort dataframe by predefined policy. Returns df and applied ops."""
    if not sort_order:
        return df, []

    key = sort_order.strip().lower()
    sort_map: Dict[str, Tuple[List[str], List[bool]]] = {
        "alphabetical": (["title"], [True]),
        "price": (["price", "title"], [True, True]),
        "rating": (["rating_rate", "rating_count", "title"], [False, False, True]),
        "category": (["category", "title"], [True, True]),
    }
    if key not in sort_map:
        raise ValueError(
            "`sort_order` must be one of: alphabetical, price, rating, category."
        )

    by, asc = sort_map[key]
    out = df.copy()
    # Fill NA in sort keys to avoid errors; keep a stable mergesort
    fill_map = {
        "title": "",
        "price": 0.0,
        "rating_rate": -1.0,
        "rating_count": -1,
        "category": "",
    }
    for col in by:
        if col in out.columns:
            out[col] = out[col].fillna(fill_map.get(col, ""))

    out = out.sort_values(by=by, ascending=asc, kind="mergesort")
    return out, [f"sort_order={key}"]


# --------------------------- Public API --------------------------------------


def edit_products_csv(
    input_csv: str,
    output_name: str,
    *,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    dedupe_on: Optional[List[str]] = None,
    sort_order: Optional[
        str
    ] = None,  # "alphabetical" | "price" | "rating" | "category"
    file_format: str = "csv",  # "csv" (default) or "xlsx"
    encoding: str = "utf-8-sig",  # Excel-friendly UTF-8
    delimiter: str = ",",  # CSV delimiter
) -> Dict[str, Any]:
    """Edit a product CSV: filter, select columns, deduplicate, sort, and save.

    Parameters
    ----------
    input_csv : str
        Path to the source CSV file to read.
    output_name : str
        Base file name (NO PATHS). The extension is forced according to `file_format`.
    include_categories : list[str], optional
        Keep only rows whose 'category' is in this list (case-insensitive).
    exclude_categories : list[str], optional
        Remove rows whose 'category' is in this list (case-insensitive).
    min_price, max_price : float, optional
        Keep rows within the price range [min_price, max_price].
    min_rating : float, optional
        Keep rows with rating_rate >= min_rating.
    keep_columns : list[str], optional
        If provided, keep only these columns (intersection with existing columns).
    drop_columns : list[str], optional
        Columns to drop (if present).
    dedupe_on : list[str], optional
        Subset of columns to deduplicate by. If None, deduplicate entire rows.
    sort_order : str, optional
        One of: 'alphabetical', 'price', 'rating', 'category'.
    file_format : str, optional
        'csv' (default) or 'xlsx'.
    encoding : str, optional
        Text encoding for CSV. Default 'utf-8-sig' is Excel-friendly.
    delimiter : str, optional
        CSV delimiter. Default ','.

    Returns
    -------
    dict
        {
          "path": absolute output path,
          "rows_before": int,
          "rows_after": int,
          "columns": list[str],
          "applied": list[str]   # human-readable list of applied operations
        }

    Raises
    ------
    FileNotFoundError
        If `input_csv` does not exist.
    ValueError
        If parameters are invalid or `output_name` is unsafe.
    """
    # --------- Validate inputs ----------
    if not isinstance(input_csv, str) or not input_csv.strip():
        raise ValueError("`input_csv` must be a non-empty path to a CSV file.")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Read CSV
    try:
        df = pd.read_csv(input_csv, encoding=encoding, sep=delimiter)
    except Exception as e:
        raise ValueError(f"Failed to read CSV '{input_csv}': {e}") from e

    rows_before = int(df.shape[0])

    # Normalize schema (rating columns)
    df = _normalize(df)

    applied_ops: List[str] = []

    # Apply filters
    df, ops = _apply_filters(
        df,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
    )
    applied_ops.extend(ops)

    # Column selection / dropping
    df, ops = _apply_column_selection(
        df, keep_columns=keep_columns, drop_columns=drop_columns
    )
    applied_ops.extend(ops)

    # Deduplicate
    df, ops = _apply_deduplication(df, dedupe_on=dedupe_on)
    applied_ops.extend(ops)

    # Sorting
    df, ops = _apply_sort(df, sort_order=sort_order)
    applied_ops.extend(ops)

    # Output file path (safe)
    safe_name = _safe_filename(output_name, file_format)
    out_path = os.path.abspath(safe_name)

    # Write output
    try:
        if file_format.lower() == "csv":
            df.to_csv(out_path, index=False, encoding=encoding)
        else:
            with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
                sheet = "Edited"
                df.to_excel(writer, index=False, sheet_name=sheet)
                # Best-effort column widths
                try:
                    ws = writer.sheets[sheet]
                    for i, col in enumerate(df.columns):
                        width = min(
                            60, max(10, int(df[col].astype(str).str.len().mean() + 2))
                        )
                        ws.set_column(i, i, width)
                except Exception:
                    pass
    except Exception as e:
        raise ValueError(f"Failed to write output '{out_path}': {e}") from e

    return {
        "path": out_path,
        "rows_before": rows_before,
        "rows_after": int(df.shape[0]),
        "columns": list(df.columns),
        "applied": applied_ops,
    }
