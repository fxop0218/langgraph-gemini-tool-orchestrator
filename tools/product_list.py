# -*- coding: utf-8 -*-
"""
Export utility for product catalogs — safe filename + single-function API.

Designed for LLM orchestrators:
- Deterministic sorting (stable 'mergesort').
- Clear, catchable exceptions with helpful messages.
- Minimal surface: one public function `export_products`.

Requires: pandas, xlsxwriter (if using xlsx).
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import os
import re
import pandas as pd


def export_products(
    payload: List[Dict[str, Any]],
    order_by: str,
    file_name: str,
    file_format: str = "xlsx",
) -> str:
    """Export a product list to CSV/Excel with a chosen sorting policy.

    Parameters
    ----------
    payload : List[Dict[str, Any]]
        Raw list of product dicts. Expected keys include:
        - id, title, price, description, category, image, rating.rate, rating.count
        Missing keys are tolerated; absent rating fields are set to NA.
    order_by : str
        Sorting policy (case-insensitive). Supported values:
        - "alphabetical": by title A→Z
        - "price": by price ↑ then title A→Z
        - "rating": by rating_rate ↓ then rating_count ↓ then title A→Z
        - "category": by category A→Z then title A→Z
    file_name : str
        Output base name (NO PATHS ALLOWED). Extension is enforced by file_format.
        Examples: "products_sorted", "catalog".
    file_format : str, optional
        "csv" or "xlsx" (default "xlsx"). Determines file extension and writer.

    Returns
    -------
    str
        Absolute path to the created file.

    Raises
    ------
    ValueError
        If inputs are invalid (unsupported order_by, unsafe file_name, etc.).
    """
    # ---------- Validate inputs ----------
    if not isinstance(payload, list):
        raise ValueError("`payload` must be a list of dicts.")

    order_key = (order_by or "").strip().lower()
    valid_orders = {"alphabetical", "price", "rating", "category"}
    if order_key not in valid_orders:
        raise ValueError(f"`order_by` must be one of {sorted(valid_orders)}.")

    fmt = (file_format or "").strip().lower()
    if fmt not in {"csv", "xlsx"}:
        raise ValueError("`file_format` must be 'csv' or 'xlsx'.")

    safe_name = _safe_filename(file_name, fmt)

    # ---------- Normalize / flatten ----------
    # Use pandas.json_normalize to flatten rating fields into rating_rate, rating_count
    df = pd.json_normalize(payload, sep="_")

    # Ensure rating columns exist even if rating is missing in some rows
    if "rating_rate" not in df.columns:
        # If the nested keys are present as rating.rate, rename; else create NA
        if "rating.rate" in df.columns:
            df = df.rename(columns={"rating.rate": "rating_rate"})
        else:
            df["rating_rate"] = pd.NA
    if "rating_count" not in df.columns:
        if "rating.count" in df.columns:
            df = df.rename(columns={"rating.count": "rating_count"})
        else:
            df["rating_count"] = pd.NA

    # Preferred column order for readability if present
    preferred = [
        "id",
        "title",
        "category",
        "price",
        "rating_rate",
        "rating_count",
        "description",
        "image",
    ]
    ordered_cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    df = df[ordered_cols]

    # ---------- Sorting policies ----------
    # Stable mergesort → deterministic results
    sort_map: Dict[str, Tuple[list, list]] = {
        "alphabetical": (["title"], [True]),
        "price": (["price", "title"], [True, True]),
        "rating": (["rating_rate", "rating_count", "title"], [False, False, True]),
        "category": (["category", "title"], [True, True]),
    }

    by, asc = sort_map[order_key]
    # Make sure missing values don't crash sorting: fill NAs in keys (without mutating source columns)
    tmp_df = df.copy()
    fill_map = {
        "title": "",
        "price": float("inf") if order_key == "price" else 0.0,
        "rating_rate": -1.0,
        "rating_count": -1,
        "category": "",
    }
    tmp_df[by] = tmp_df[by].apply(lambda s: s.fillna(fill_map.get(s.name, "")))

    sorted_df = tmp_df.sort_values(by=by, ascending=asc, kind="mergesort")

    # ---------- Write output ----------
    out_path = os.path.abspath(safe_name)
    if fmt == "csv":
        sorted_df.to_csv(out_path, index=False)
    else:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            # Single sheet to keep API simple for orchestrators
            sheet = _sheet_name_for(order_key)
            sorted_df.to_excel(writer, index=False, sheet_name=sheet)

            # Optional niceties: set column widths (best-effort; no errors if it fails)
            try:
                ws = writer.sheets[sheet]
                for i, col in enumerate(sorted_df.columns):
                    # Cap width to 60 to avoid huge columns
                    width = min(
                        60,
                        max(10, int(sorted_df[col].astype(str).str.len().mean() + 2)),
                    )
                    ws.set_column(i, i, width)
            except Exception:
                pass  # Non-fatal

    return out_path


# ------------------------- Helpers (internal) -------------------------

_SAFE_NAME_RE = re.compile(
    r"^[\w\-. ]+$"
)  # letters, digits, underscore, dash, dot, space


def _safe_filename(name: str, fmt: str) -> str:
    """Return a sanitized base filename with the right extension, no paths allowed.

    - Rejects path separators (/ or \\), drive prefixes, and parent traversals.
    - Restricts characters to letters, digits, underscore, dash, dot, and space.
    - Forces extension based on `fmt`.

    Examples:
        _safe_filename("catalog", "csv")      -> "catalog.csv"
        _safe_filename("my export", "xlsx")   -> "my export.xlsx"
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`file_name` must be a non-empty string.")

    raw = name.strip()

    # Disallow any path-like patterns
    if os.path.basename(raw) != raw:
        raise ValueError("`file_name` must not contain path separators.")
    if raw.startswith((".", "~")) or ".." in raw:
        raise ValueError("`file_name` must not start with '.' or '~' or contain '..'.")
    if "/" in raw or "\\" in raw or ":" in raw:
        raise ValueError("`file_name` must not contain '/', '\\\\', or ':' characters.")

    # Restrict character set
    if not _SAFE_NAME_RE.match(raw):
        raise ValueError("`file_name` contains unsupported characters.")

    # Normalize extension to match format
    ext = ".csv" if fmt == "csv" else ".xlsx"
    base = raw
    if base.lower().endswith((".csv", ".xlsx")):
        base = base[: base.lower().rfind(".")]  # drop existing extension

    return f"{base}{ext}"


def _sheet_name_for(order_key: str) -> str:
    """Return a short, Excel-safe sheet name for the given order key."""
    mapping = {
        "alphabetical": "Alphabetical",
        "price": "By Price",
        "rating": "By Rating",
        "category": "By Category",
    }
    # Excel sheet name limit is 31 chars; these are short already
    return mapping.get(order_key, "Data")


# ------------------------- Example (commented) -------------------------
# Example usage within an orchestrator tool call:
#
# try:
#     path = export_products(
#         payload=products_json,          # ← list[dict] as in your example
#         order_by="rating",              # "alphabetical" | "price" | "rating" | "category"
#         file_name="products_sorted",    # ← ONLY a base name, no paths
#         file_format="csv",              # "csv" | "xlsx"
#     )
#     # Return `path` to the calling agent/tool
# except ValueError as e:
#     # Surface the error message to the LLM/tooling layer
#     err_msg = f"export_products_error: {e}"
#     ...
