import pandas as pd

from src.pipeline.eia_client import DAILY_SUB_BA_ENDPOINT, SUB_BA_ENDPOINT, EIAClient


def test_daily_endpoint_request_params() -> None:
    client = EIAClient(base_url="https://api.eia.gov/v2", api_key="KEY")
    params = client._params_for_request(
        endpoint=DAILY_SUB_BA_ENDPOINT,
        regions=["CISO"],
        start=pd.Timestamp("2024-01-01T00:00:00Z"),
        end=pd.Timestamp("2024-02-01T00:00:00Z"),
        offset=0,
        length=5000,
        extra_facets=[("timezone", "Eastern")],
    )
    params_dict = dict(params)
    assert params_dict["frequency"] == "daily"
    # Daily periods are formatted as plain dates (no hour component).
    assert params_dict["start"] == "2024-01-01"
    assert params_dict["end"] == "2024-02-01"
    assert ("facets[parent][]", "CISO") in params
    assert ("facets[timezone][]", "Eastern") in params


def test_hourly_endpoint_request_params() -> None:
    client = EIAClient(base_url="https://api.eia.gov/v2", api_key="KEY")
    params = dict(
        client._params_for_request(
            endpoint=SUB_BA_ENDPOINT,
            regions=["CISO"],
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-02T00:00:00Z"),
            offset=0,
            length=5000,
        )
    )
    assert params["frequency"] == "hourly"
    assert params["start"] == "2024-01-01T00"
