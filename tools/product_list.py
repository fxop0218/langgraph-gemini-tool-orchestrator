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
    """Export a product list to CSV/Excel with a chosen sorting policy."""
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

    df = pd.json_normalize(payload, sep="_")

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

    sort_map: Dict[str, Tuple[list, list]] = {
        "alphabetical": (["title"], [True]),
        "price": (["price", "title"], [True, True]),
        "rating": (["rating_rate", "rating_count", "title"], [False, False, True]),
        "category": (["category", "title"], [True, True]),
    }

    by, asc = sort_map[order_key]
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

    out_path = os.path.abspath(safe_name)
    if fmt == "csv":
        sorted_df.to_csv(out_path, index=False)
    else:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            sheet = _sheet_name_for(order_key)
            sorted_df.to_excel(writer, index=False, sheet_name=sheet)
            try:
                ws = writer.sheets[sheet]
                for i, col in enumerate(sorted_df.columns):
                    width = min(
                        60,
                        max(10, int(sorted_df[col].astype(str).str.len().mean() + 2)),
                    )
                    ws.set_column(i, i, width)
            except Exception:
                pass
    return out_path


_SAFE_NAME_RE = re.compile(r"^[\w\-. ]+$")


def _safe_filename(name: str, fmt: str) -> str:
    """Return a sanitized base filename with the right extension, no paths allowed."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`file_name` must be a non-empty string.")

    raw = name.strip()

    if os.path.basename(raw) != raw:
        raise ValueError("`file_name` must not contain path separators.")
    if raw.startswith((".", "~")) or ".." in raw:
        raise ValueError("`file_name` must not start with '.' or '~' or contain '..'.")
    if "/" in raw or "\\" in raw or ":" in raw:
        raise ValueError("`file_name` must not contain '/', '\\\\', or ':' characters.")

    if not _SAFE_NAME_RE.match(raw):
        raise ValueError("`file_name` contains unsupported characters.")

    ext = ".csv" if fmt == "csv" else ".xlsx"
    base = raw
    if base.lower().endswith((".csv", ".xlsx")):
        base = base[: base.lower().rfind(".")]
    return f"{base}{ext}"


def _sheet_name_for(order_key: str) -> str:
    """Return a short, Excel-safe sheet name for the given order key."""
    mapping = {
        "alphabetical": "Alphabetical",
        "price": "By Price",
        "rating": "By Rating",
        "category": "By Category",
    }
    return mapping.get(order_key, "Data")
