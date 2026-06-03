from __future__ import annotations

import pandas as pd


def ensure_datetime_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt


def transform_subba_to_region_total(df: pd.DataFrame, allowed_regions: tuple[str, ...]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["region", "series", "period", "value"])

    out = df.copy()
    out["period"] = ensure_datetime_utc(out["period"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    out["parent"] = out["parent"].astype(str).str.upper()
    out = out[out["parent"].isin(allowed_regions)]
    out = out.dropna(subset=["period", "value"])

    grouped = (
        out.groupby(["parent", "period"], as_index=False)["value"]
        .sum()
        .rename(columns={"parent": "region"})
    )
    grouped["series"] = grouped["region"] + " total"
    grouped = grouped[["region", "series", "period", "value"]].sort_values(["region", "period"])
    return grouped.reset_index(drop=True)


def deduplicate_latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
