"""
tests/test_analyze_file_endpoint.py
------------------------------------
STAGE 13/14: a genuine END-TO-END integration test of the real HTTP
endpoint the frontend calls -- POST /analyze-file with an actual real
image file, using FastAPI's TestClient (which runs the real ASGI app,
the real route function, the real image_extractor.extract_csr(), and
the real analysis/salience/narration pipeline -- nothing here is
mocked or stubbed). This is the automated version of "start the
backend, send it revenue.jpg with curl" from the integration brief,
so it can be re-run in CI instead of only by hand.
"""

import os

from fastapi.testclient import TestClient

from api import app

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CHART = os.path.join(FIXTURE_DIR, "sample_line_chart.png")

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_real_png_upload_produces_a_real_analysis_response():
    with open(SAMPLE_CHART, "rb") as f:
        response = client.post(
            "/analyze-file",
            files={"file": ("revenue.png", f, "image/png")},
        )

    assert response.status_code == 200, response.text
    body = response.json()

    # This is the actual contract the frontend depends on -- assert its
    # real shape, not just "some 200 came back."
    assert body["chart_id"] == "revenue"
    assert "analysis" in body and "series" in body["analysis"]
    assert len(body["analysis"]["series"]) == 1
    series = body["analysis"]["series"][0]
    assert series["overall_trend"]["direction"] in ("increasing", "decreasing", "flat")

    assert "salience_summaries" in body and len(body["salience_summaries"]) == 1
    assert "narrations" in body and len(body["narrations"]) == 1
    narration_text = body["narrations"][0]["text"]
    assert isinstance(narration_text, str) and len(narration_text) > 0
    # The narration must be grounded in the real extracted/analyzed
    # numbers -- never a canned/fake string.
    assert body["narrations"][0]["grounded"] is True


def test_real_jpg_upload_end_to_end():
    """The exact scenario from the integration brief: an actual .jpg
    (not .png) travels from a client, through multipart/form-data, into
    the real OCR/extraction pipeline, and back out as real analysis."""
    from PIL import Image
    import io

    img = Image.open(SAMPLE_CHART).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    response = client.post(
        "/analyze-file",
        files={"file": ("revenue.jpg", buf, "image/jpeg")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["analysis"]["series"]) == 1, (
        "a real JPG upload must still resolve to exactly one series "
        "(regression check for the hue-clustering JPEG artifact bug)"
    )


def test_empty_file_returns_clean_422_not_a_crash():
    response = client.post(
        "/analyze-file",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_corrupted_image_returns_clean_422_not_a_stack_trace():
    response = client.post(
        "/analyze-file",
        files={"file": ("broken.png", b"not a real image", "image/png")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert "corrupted" in detail.lower() or "valid image" in detail.lower()


def test_unsupported_extension_returns_415_with_helpful_message():
    response = client.post(
        "/analyze-file",
        files={"file": ("chart.pdf", b"%PDF-fake", "application/pdf")},
    )
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_unrecognized_extension_returns_415():
    response = client.post(
        "/analyze-file",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_malformed_json_upload_returns_clean_422():
    response = client.post(
        "/analyze-file",
        files={"file": ("chart.json", b"{not valid json", "application/json")},
    )
    assert response.status_code == 422
    assert "json" in response.json()["detail"].lower()


def test_valid_json_csr_upload_is_analyzed_directly():
    import json

    csr_json = json.dumps({
        "chart_id": "uploaded",
        "title": "Uploaded Chart",
        "chart_type": "line",
        "x_axis": {"label": "Month", "unit": None, "values": ["Jan", "Feb", "Mar"]},
        "y_axis": {"label": "Value", "unit": None, "min": None, "max": None},
        "series": [{
            "id": "s1", "name": "Value",
            "points": [{"x": "Jan", "y": 10}, {"x": "Feb", "y": 20}, {"x": "Mar", "y": 15}],
        }],
    }).encode("utf-8")

    response = client.post(
        "/analyze-file",
        files={"file": ("chart.json", csr_json, "application/json")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["chart_id"] == "uploaded"
