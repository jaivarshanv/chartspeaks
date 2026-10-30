"""
tests/test_bar_charts.py
-------------------------
Real tests for bar-chart images, run against actual matplotlib-rendered
PNGs in tests/fixtures/ (not hand-typed CSRs).

The regression these lock in: bar charts used to go through the same
column sampler as lines, which takes the MEAN height of a column's
colored pixels. On a bar that is the bar's middle, so every value came
back at half its real size -- with full confidence, and no error.

Ground truth for every fixture: Jan..Jun on the X axis.
  bar_chart.png            12/18/9/25/21/15, one color, grid behind bars
  bar_chart_multicolor.png the same values, a different color per bar
  bar_chart_grouped.png    the same values side by side with 8/14/16/11/19/22
  bar_chart_negative.png   5/-3/8/-6/2/4
  bar_chart_stacked.png    two stacked series -- must be rejected, not misread
"""

import io
import os

import pytest
from fastapi.testclient import TestClient

from api import app
from image_extractor import extract_csr, ExtractionError

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
SALES = [12, 18, 9, 25, 21, 15]

client = TestClient(app)


def _read(name: str) -> bytes:
    with open(os.path.join(FIXTURE_DIR, name), "rb") as f:
        return f.read()


def _assert_values(series, expected):
    points = {p.x: p.y for p in series.points}
    assert list(points) == LABELS
    for label, true_value in zip(LABELS, expected):
        # Real OCR + pixel measurement: allow 2% of the axis range.
        assert abs(points[label] - true_value) < 0.6, (
            f"{label}: extracted {points[label]}, expected ~{true_value}"
        )


@pytest.mark.parametrize("fixture", ["bar_chart.png", "bar_chart_multicolor.png"])
def test_bar_values_are_read_from_the_top_of_each_bar(fixture):
    csr = extract_csr(_read(fixture), chart_id="bars")

    assert csr.chart_type == "bar"
    assert len(csr.series) == 1, [s.name for s in csr.series]
    _assert_values(csr.series[0], SALES)


def test_bar_chart_survives_jpeg_recompression():
    from PIL import Image

    buffer = io.BytesIO()
    Image.open(os.path.join(FIXTURE_DIR, "bar_chart.png")).convert("RGB").save(buffer, "JPEG", quality=90)

    csr = extract_csr(buffer.getvalue(), chart_id="bars_jpg")
    assert csr.chart_type == "bar"
    assert len(csr.series) == 1
    _assert_values(csr.series[0], SALES)


def test_grouped_bars_become_one_series_per_color():
    csr = extract_csr(_read("bar_chart_grouped.png"), chart_id="grouped")

    assert csr.chart_type == "bar"
    assert len(csr.series) == 2
    _assert_values(csr.series[0], SALES)
    _assert_values(csr.series[1], [8, 14, 16, 11, 19, 22])


def test_negative_bars_are_read_from_their_bottom_edge():
    csr = extract_csr(_read("bar_chart_negative.png"), chart_id="profit")

    assert csr.chart_type == "bar"
    _assert_values(csr.series[0], [5, -3, 8, -6, 2, 4])


def test_stacked_bars_are_rejected_instead_of_misread():
    with pytest.raises(ExtractionError, match="stacked"):
        extract_csr(_read("bar_chart_stacked.png"), chart_id="stacked")


def test_bar_upload_end_to_end_reports_a_bar_chart():
    response = client.post(
        "/analyze-file",
        files={"file": ("monthly_sales.png", _read("bar_chart.png"), "image/png")},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # The echoed chart is what lets the frontend draw real bars.
    assert body["chart"]["chart_type"] == "bar"
    assert body["chart"]["title"] == "Monthly Sales"
    assert [p["x"] for p in body["chart"]["series"][0]["points"]] == LABELS

    maximum = body["analysis"]["series"][0]["range"]["maximum"]
    assert maximum["x"] == "Apr" and abs(maximum["y"] - 25) < 0.6

    text = body["narrations"][0]["text"]
    assert "bar chart" in text
    # No holes where a missing axis label would have gone.
    assert " ," not in text and "the  " not in text
