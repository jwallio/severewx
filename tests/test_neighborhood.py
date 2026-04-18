import numpy as np

from severewx.labels.neighborhood import neighborhood_hits


def test_neighborhood_hits_marks_nearby_gridpoints() -> None:
    lat = np.array([[35.0, 35.0], [36.0, 36.0]])
    lon = np.array([[-98.0, -97.0], [-98.0, -97.0]])
    report_lat = np.array([35.1])
    report_lon = np.array([-97.9])
    hits = neighborhood_hits(lat, lon, report_lat, report_lon, radius_km=30.0)
    assert hits.shape == lat.shape
    assert hits[0, 0] == 1
    assert hits[1, 1] == 0
