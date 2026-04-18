import warnings

import pandas as pd

from severewx.config import load_settings
from severewx.features.analogs import build_analog_reference


def test_build_analog_reference_all_na_outbreak_modes_uses_none_without_futurewarning() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "date": ["2024-04-27"],
            "lead_day": [1],
            "cape": [1500.0],
            "tornado_outbreak": [float("nan")],
            "hail_outbreak": [float("nan")],
            "wind_outbreak": [float("nan")],
            "any_outbreak": [float("nan")],
            "significant_tornado_support": [float("nan")],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        reference = build_analog_reference(frame, settings)
    assert not any(isinstance(item.message, FutureWarning) for item in caught)
    assert reference.loc[0, "dominant_mode"] == "none"
