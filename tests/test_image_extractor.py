"""
tests/test_image_extractor.py
------------------------------
STAGE 14: real tests for image_extractor.py -- run against an ACTUAL
image file on disk (tests/fixtures/sample_line_chart.png, the same
sample chart already bundled with this project's data/inputs/), not a
hand-typed fake CSR. This is the "does the digitizer actually read a
real chart correctly" check the integration brief asked for.

Ground truth for the fixture image: a 5-point line-with-markers chart,
Mon/Tues/Wed/Thurs/Fri on the X axis, with values 300/450/200/400/650.
We assert the extracted values land close to those (OCR + pixel
measurement is never pixel-perfect -- these are real measurements off a
real image, not fabricated numbers, so a small tolerance is expected
and honest), rather than asserting exact equality.
"""

import os

import pytest

from image_extractor import extract_csr, ExtractionError

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CHART = os.path.join(FIXTURE_DIR, "sample_line_chart.png")


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_extracts_correct_single_series_from_real_chart_image():
    csr = extract_csr(_read(SAMPLE_CHART), chart_id="sample_line_chart")

    assert csr.chart_type == "line"
    assert len(csr.series) == 1, (
        f"expected exactly one real series, got {len(csr.series)} "
        f"({[s.name for s in csr.series]}) -- markers/line were not "
        f"correctly recognized as the same series"
    )

    points = {p.x: p.y for p in csr.series[0].points}
    expected = {"Mon": 300, "Tues": 450, "Wed": 200, "Thurs": 400, "Fri": 650}
    assert set(points.keys()) == set(expected.keys())
    for label, true_value in expected.items():
        # Real OCR + pixel measurement, not exact -- allow ~5% of the
        # chart's value range as honest slack.
        assert abs(points[label] - true_value) < 30, (
            f"{label}: extracted {points[label]}, expected ~{true_value}"
        )

    # Confidence scores must be genuine (derived from measured pixel
    # alignment), not a hardcoded 1.0 for every point.
    confidences = [p.confidence for p in csr.series[0].points]
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert len(set(confidences)) > 1, "confidences look hardcoded, not measured"


def test_survives_jpeg_recompression_of_the_same_chart(tmp_path):
    """The exact regression this test locks in: re-saving the sample
    chart as a lossy JPEG (as any real phone/browser upload would be)
    must still produce ONE series, not several near-duplicate ones from
    hue-clustering artifacts introduced by compression."""
    from PIL import Image

    jpg_path = tmp_path / "sample_line_chart.jpg"
    Image.open(SAMPLE_CHART).convert("RGB").save(jpg_path, quality=95)

    csr = extract_csr(_read(str(jpg_path)), chart_id="sample_line_chart_jpg")
    assert len(csr.series) == 1, (
        f"JPEG recompression produced {len(csr.series)} series instead of 1 "
        f"-- likely the hue-clustering circular-mean regression"
    )
    assert len(csr.series[0].points) == 5


def test_garbage_bytes_raise_a_clear_extraction_error_not_a_crash():
    with pytest.raises(ExtractionError):
        extract_csr(b"this is not an image", chart_id="garbage")


def test_blank_image_with_no_axis_labels_raises_extraction_error():
    import numpy as np
    import cv2

    blank = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(blank, (20, 20), (280, 180), (0, 0, 0), 2)
    ok, encoded = cv2.imencode(".png", blank)
    assert ok
    with pytest.raises(ExtractionError):
        extract_csr(encoded.tobytes(), chart_id="blank")
