from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Iterable
import os
import re
import pandas as pd

__all__ = ["edit_products_excel"]

_SAFE_NAME_RE = re.compile(r"^[\w\-. ]+$")


def _safe_filename(name: str) -> str:
    """Return a sanitized base filename with .xlsx extension, no paths allowed."""
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

    base = raw
    if base.lower().endswith(".xlsx"):
        base = base[:-5]
    return f"{base}.xlsx"


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
    return _normalize_rating_columns(df)


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
        out = out[cat_series.str.lower().isin(include_set)]
        applied.append(f"include_categories={sorted(include_set)}")

    if exclude_categories:
        exclude_set = {c.strip().lower() for c in exclude_categories if str(c).strip()}
        out = out[~cat_series.str.lower().isin(exclude_set)]
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


def _remove_rows_by_ids(
    df: pd.DataFrame, ids: Optional[Iterable]
) -> tuple[pd.DataFrame, list[str]]:
    if not ids:
        return df, []
    if "id" not in df.columns:
        return df, ["remove_by_ids=skipped(no 'id' column)"]
    ids_set = set(ids)
    out = df[~df["id"].isin(ids_set)]
    removed = int(len(df) - len(out))
    return out, [f"remove_by_ids={removed}"]


def _remove_where_equals(
    df: pd.DataFrame, rules: Optional[dict[str, list]]
) -> tuple[pd.DataFrame, list[str]]:
    if not rules:
        return df, []
    out = df.copy()
    applied = []
    for col, vals in rules.items():
        if col not in out.columns or not isinstance(vals, (list, tuple, set)):
            continue
        series = out[col]
        if series.dtype == object:
            mask = series.astype(str).str.lower().isin({str(v).lower() for v in vals})
        else:
            mask = series.isin(vals)
        before = len(out)
        out = out[~mask]
        applied.append(f"remove_equals[{col}]={before-len(out)}")
    return out, applied


def _remove_where_contains(
    df: pd.DataFrame, rules: Optional[dict[str, list[str]]]
) -> tuple[pd.DataFrame, list[str]]:
    if not rules:
        return df, []
    out = df.copy()
    applied = []
    for col, substrings in rules.items():
        if col not in out.columns or not substrings:
            continue
        patt = "|".join([re.escape(s) for s in substrings if str(s).strip()])
        if not patt:
            continue
        mask = out[col].astype(str).str.contains(patt, case=False, na=False)
        before = len(out)
        out = out[~mask]
        applied.append(f"remove_contains[{col}]={before-len(out)}")
    return out, applied


def _remove_where_regex(
    df: pd.DataFrame, rules: Optional[dict[str, str]]
) -> tuple[pd.DataFrame, list[str]]:
    if not rules:
        return df, []
    out = df.copy()
    applied = []
    for col, pattern in rules.items():
        if (
            col not in out.columns
            or not isinstance(pattern, str)
            or not pattern.strip()
        ):
            continue
        mask = (
            out[col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
        )
        before = len(out)
        out = out[~mask]
        applied.append(f"remove_regex[{col}]={before-len(out)}")
    return out, applied


def _remove_rows_with_nulls(
    df: pd.DataFrame, columns: Optional[list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    if not columns:
        return df, []
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df, ["remove_nulls=skipped(no columns)"]
    before = len(df)
    out = df.dropna(subset=cols)
    return out, [f"remove_nulls[{','.join(cols)}]={before-len(out)}"]


def _remove_where_zero(
    df: pd.DataFrame, columns: Optional[list[str]]
) -> tuple[pd.DataFrame, list[str]]:
    if not columns:
        return df, []
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df, ["remove_zero=skipped(no columns)"]
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | (pd.to_numeric(df[c], errors="coerce") == 0)
    before = len(df)
    out = df[~mask]
    return out, [f"remove_zero[{','.join(cols)}]={before-len(out)}"]


def _safe_input_name(name: str, exts: tuple[str, ...] = (".xlsx",)) -> str:
    """Sanitize an input file name (no paths) and ensure valid extension."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("`input_name` must be a non-empty string.")
    raw = name.strip()

    if os.path.basename(raw) != raw:
        raise ValueError("`input_name` must not contain path separators.")
    if raw.startswith((".", "~")) or ".." in raw:
        raise ValueError("`input_name` must not start with '.' or '~' or contain '..'.")
    if "/" in raw or "\\" in raw or ":" in raw:
        raise ValueError(
            "`input_name` must not contain '/', '\\\\', or ':' characters."
        )
    if not _SAFE_NAME_RE.match(raw):
        raise ValueError("`input_name` contains unsupported characters.")

    lowered = raw.lower()
    if not any(lowered.endswith(ext) for ext in exts):
        raw = f"{raw}{exts[0]}"
    return raw


def edit_products_excel(
    input_excel: Optional[str] = None,
    output_name: Optional[str] = None,
    *,
    input_name: Optional[str] = None,
    in_place: bool = False,
    sheet_name: Optional[str] = None,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    keep_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
    dedupe_on: Optional[List[str]] = None,
    sort_order: Optional[str] = None,
    output_sheet: str = "Edited",
    engine_read: Optional[str] = None,
    remove_ids: Optional[List] = None,
    remove_equals: Optional[Dict[str, List]] = None,
    remove_contains: Optional[Dict[str, List[str]]] = None,
    remove_regex: Optional[Dict[str, str]] = None,
    remove_nulls_in: Optional[List[str]] = None,
    remove_zero_in: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if input_name:
        safe_in = _safe_input_name(input_name, (".xlsx",))
        base_dir = os.path.abspath(os.path.join("files", "excel"))
        os.makedirs(base_dir, exist_ok=True)
        input_path = os.path.join(base_dir, safe_in)
    else:
        if not isinstance(input_excel, str) or not input_excel.strip():
            raise ValueError("Provide `input_name` or a valid `input_excel` path.")
        input_path = input_excel

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input Excel not found: {input_path}")

    try:
        df = pd.read_excel(input_path, sheet_name=sheet_name, engine=engine_read)
    except Exception as e:
        raise ValueError(f"Failed to read Excel '{input_path}': {e}") from e

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

    df, ops = _remove_rows_by_ids(df, remove_ids)
    applied_ops.extend(ops)

    df, ops = _remove_where_equals(df, remove_equals)
    applied_ops.extend(ops)

    df, ops = _remove_where_contains(df, remove_contains)
    applied_ops.extend(ops)

    df, ops = _remove_where_regex(df, remove_regex)
    applied_ops.extend(ops)

    df, ops = _remove_rows_with_nulls(df, remove_nulls_in)
    applied_ops.extend(ops)

    df, ops = _remove_where_zero(df, remove_zero_in)
    applied_ops.extend(ops)

    df, ops = _apply_column_selection(
        df, keep_columns=keep_columns, drop_columns=drop_columns
    )
    applied_ops.extend(ops)

    df, ops = _apply_deduplication(df, dedupe_on=dedupe_on)
    applied_ops.extend(ops)

    df, ops = _apply_sort(df, sort_order=sort_order)
    applied_ops.extend(ops)

    if in_place:
        base_dir = os.path.dirname(input_path) or "."
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.basename(input_path)
        tmp_path = os.path.abspath(os.path.join(base_dir, f".__tmp__{base_name}"))
        final_path = input_path
    else:
        if not output_name:
            raise ValueError("`output_name` is required when `in_place=False`.")
        safe_out = _safe_filename(output_name)
        base_dir = os.path.abspath(os.path.join("files", "excel"))
        os.makedirs(base_dir, exist_ok=True)
        final_path = os.path.join(base_dir, safe_out)
        tmp_path = final_path

    try:
        with pd.ExcelWriter(tmp_path, engine="xlsxwriter") as writer:
            sheet = (output_sheet or "Edited")[:31] or "Edited"
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
        if in_place and tmp_path != final_path:
            os.replace(tmp_path, final_path)
    except Exception as e:
        raise ValueError(f"Failed to write output '{final_path}': {e}") from e

    return {
        "path": final_path,
        "rows_before": rows_before,
        "rows_after": int(df.shape[0]),
        "columns": list(df.columns),
        "applied": applied_ops,
        "in_place": bool(in_place),
    }
