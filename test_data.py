"""
test_data.py
------------
This file holds FAKE chart data so we can build and test the analysis
engine before Person 1's real Chart4Blind pipeline is ready.

The dictionary below matches the CSR schema exactly (see models.py).
Later, when Person 1's code is ready, their output JSON gets loaded the
same way get_fake_csr() loads this one -- nothing else in the project
needs to change.
"""

from models import ChartCSR

# This is our one and only test chart for Stage 1:
# Jan=80, Feb=75, Mar=72, Apr=70, May=95 (spike), Jun=60 (crash)
FAKE_CSR_DICT = {
    "chart_id": "example_001",
    "title": "Monthly Energy Consumption",
    "chart_type": "line",
    "x_axis": {
        "label": "Month",
        "unit": None,
        "values": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    },
    "y_axis": {
        "label": "Energy Consumption",
        "unit": "kWh",
        "min": 0,
        "max": 100,
    },
    "series": [
        {
            "id": "series_1",
            "name": "Consumption",
            "points": [
                {"x": "Jan", "y": 80, "confidence": 0.98},
                {"x": "Feb", "y": 75, "confidence": 0.97},
                {"x": "Mar", "y": 72, "confidence": 0.96},
                {"x": "Apr", "y": 70, "confidence": 0.97},
                {"x": "May", "y": 95, "confidence": 0.94},
                {"x": "Jun", "y": 60, "confidence": 0.96},
            ],
        }
    ],
}


def get_fake_csr() -> ChartCSR:
    """
    Turns the plain Python dictionary above into a validated ChartCSR object.

    ChartCSR(**FAKE_CSR_DICT) means: "unpack this dictionary as keyword
    arguments into the ChartCSR class." Pydantic then checks every field,
    every nested object (x_axis, y_axis, series, points), and every type.
    If anything doesn't match the schema in models.py, this line will
    raise a clear error right here -- which is exactly what we want,
    instead of the bad data silently breaking analyzer.py later.
    """
    return ChartCSR(**FAKE_CSR_DICT)


# =========================================================
# STAGE 7 TEST CHART: two series that cross
# =========================================================
# Series A climbs steadily 10 -> 50. Series B falls steadily 50 -> 10.
# This is deliberately the exact example from the spec: they start on
# opposite ends, cross in the middle, and diverge again afterward.
FAKE_MULTI_SERIES_CSR_DICT = {
    "chart_id": "example_002",
    "title": "Series A vs Series B",
    "chart_type": "line",
    "x_axis": {
        "label": "Step",
        "unit": None,
        "values": ["T1", "T2", "T3", "T4", "T5"],
    },
    "y_axis": {
        "label": "Value",
        "unit": None,
        "min": 0,
        "max": 60,
    },
    "series": [
        {
            "id": "series_a",
            "name": "Series A",
            "points": [
                {"x": "T1", "y": 10, "confidence": 1.0},
                {"x": "T2", "y": 20, "confidence": 1.0},
                {"x": "T3", "y": 30, "confidence": 1.0},
                {"x": "T4", "y": 40, "confidence": 1.0},
                {"x": "T5", "y": 50, "confidence": 1.0},
            ],
        },
        {
            "id": "series_b",
            "name": "Series B",
            "points": [
                {"x": "T1", "y": 50, "confidence": 1.0},
                {"x": "T2", "y": 40, "confidence": 1.0},
                {"x": "T3", "y": 30, "confidence": 1.0},
                {"x": "T4", "y": 20, "confidence": 1.0},
                {"x": "T5", "y": 10, "confidence": 1.0},
            ],
        },
    ],
}


def get_fake_multi_series_csr() -> ChartCSR:
    """Same idea as get_fake_csr(), but loads the two-series crossing example above."""
    return ChartCSR(**FAKE_MULTI_SERIES_CSR_DICT)