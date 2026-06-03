import pandas as pd

from src.pipeline.transform import transform_subba_to_region_total


def test_transform_subba_aggregates_to_region_total() -> None:
    raw = pd.DataFrame(
        {
            "parent": ["CISO", "CISO", "PJM"],
            "subba": ["A", "B", "C"],
            "period": ["2026-01-01T00", "2026-01-01T00", "2026-01-01T01"],
            "value": [10, 15, 20],
        }
    )

    out = transform_subba_to_region_total(raw, ("CISO", "PJM"))
    assert set(out.columns) == {"region", "series", "period", "value"}
    assert str(out["period"].dtype).startswith("datetime64[ns, UTC]")

    ciso = out[(out["region"] == "CISO") & (out["period"] == pd.Timestamp("2026-01-01 00:00:00+00:00"))]
    assert float(ciso.iloc[0]["value"]) == 25.0
    assert ciso.iloc[0]["series"] == "CISO total"
