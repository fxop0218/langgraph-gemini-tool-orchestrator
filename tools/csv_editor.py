from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import re
import pandas as pd

__all__ = ["edit_products_csv"]

_SAFE_NAME_RE = re.compile(r"^[\w\-. ]+$")


def _safe_filename(name: str, fmt: str = "csv") -> str:
    """Return a sanitized base filename with the right extension, no paths allowed."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`output_name` must be a non-empty string.")

    raw = name.strip()

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

    if not _SAFE_NAME_RE.match(raw):
        raise ValueError("`output_name` contains unsupported characters.")

    fmt = (fmt or "csv").lower()
    if fmt not in {"csv", "xlsx"}:
        raise ValueError("`file_format` must be 'csv' or 'xlsx'.")

    ext = ".csv" if fmt == "csv" else ".xlsx"
    base = raw
    if base.lower().endswith((".csv", ".xlsx")):
        base = base[: base.lower().rfind(".")]

    return f"{base}{ext}"


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
    df = _normalize_rating_columns(df)
    return df


def _apply_filters(
    df: pd.DataFrame,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    applied: List[str] = []
    out = df.copy()

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
    applied: List[str] = []
    out = df.copy()

    if keep_columns:
        keep = [c for c in keep_columns if c in out.columns]
        out = out[keep] if keep else out.iloc[:, 0:0]
        applied.append(f"keep_columns={keep}")

    if drop_columns:
        drop = [c for c in drop_columns if c in out.columns]
        out = out.drop(columns=drop)
        applied.append(f"drop_columns={drop}")

    return out, applied


def _apply_deduplication(
    df: pd.DataFrame, dedupe_on: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
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
    sort_order: Optional[str] = None,
    file_format: str = "csv",
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> Dict[str, Any]:
    if not isinstance(input_csv, str) or not input_csv.strip():
        raise ValueError("`input_csv` must be a non-empty path to a CSV file.")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    try:
        df = pd.read_csv(input_csv, encoding=encoding, sep=delimiter)
    except Exception as e:
        raise ValueError(f"Failed to read CSV '{input_csv}': {e}") from e

    rows_before = int(df.shape[0])
    df = _normalize(df)
    applied_ops: List[str] = []

    df, ops = _apply_filters(
        df,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
    )
    applied_ops.extend(ops)

    df, ops = _apply_column_selection(
        df, keep_columns=keep_columns, drop_columns=drop_columns
    )
    applied_ops.extend(ops)

    df, ops = _apply_deduplication(df, dedupe_on=dedupe_on)
    applied_ops.extend(ops)

    df, ops = _apply_sort(df, sort_order=sort_order)
    applied_ops.extend(ops)

    safe_name = _safe_filename(output_name, file_format)
    if file_format.lower() == "csv":
        out_dir = os.path.abspath(os.path.join("files", "csv"))
    else:
        out_dir = os.path.abspath(os.path.join("files", "excel"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, safe_name)

    try:
        if file_format.lower() == "csv":
            df.to_csv(out_path, index=False, encoding=encoding)
        else:
            with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
                sheet = "Edited"
                df.to_excel(writer, index=False, sheet_name=sheet)
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
