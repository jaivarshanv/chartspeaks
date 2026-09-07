# ChartSpeak API Contract

This documents what the ChartSpeak backend (`api.py`) **actually exposes
today**, verified by running the server and sending it real requests
(not a hypothetical/aspirational schema). Base URL in local development:
`http://127.0.0.1:8000` (or whatever host/port you pass to `uvicorn`).

## `GET /health`

Trivial liveness check.

**Response `200`**
```json
{"status": "ok"}
```

## `POST /analyze` — analyze a chart you've already digitized

Use this when you already have a `ChartCSR` JSON object (e.g. produced
by your own tooling, or re-submitting a previously extracted chart).

- **Request**: `Content-Type: application/json`, body = a `ChartCSR` object (see schema below).
- **Response `200`**: a `ChartSpeakResponse` (see schema below).
- **Response `422`**: automatic FastAPI/Pydantic validation error if the body doesn't match `ChartCSR`.
- **Response `500`**: `{"detail": "Analysis failed: <reason>"}` if analysis itself throws unexpectedly.

## `POST /analyze-file` — analyze an uploaded file (the endpoint the frontend uses)

**This is the endpoint the upload UI should call.** It is a single
multipart-form endpoint that branches on the uploaded file's extension,
so the frontend never has to know or care which of these formats the
user picked.

- **Method**: `POST`
- **Request**: `multipart/form-data` with exactly one field:
  - `file`: the uploaded file (browser `File`/`Blob`). **Never set
    `Content-Type` manually** — let the browser/`fetch` set the
    multipart boundary itself.
- **Response `200`**: a `ChartSpeakResponse` (schema below), for every supported format.

### Supported file types

| Extension | What happens |
|---|---|
| `.png`, `.jpg`, `.jpeg` | Run through `image_extractor.extract_csr()` — a real, deterministic OpenCV + Tesseract OCR chart digitizer (**no LLM, no guessing** — see "How image extraction works" below). |
| `.json` | Parsed directly as a `ChartCSR` (`ChartCSR.model_validate(...)`). Use this if you already have a CSR, e.g. exported by another tool. |
| `.csv` | Parsed as a Chart4Blind `ExportModal` CSV export via `chart4blind_adapter.parse_chart4blind_csv()` (Chart4Blind's existing manual-digitization export format). |

### Not yet supported

`.pdf`, `.webp`, `.svg`, `.gif`, `.bmp` return a `415` with a message telling the caller to export as PNG/JPG instead, or use `.json`/`.csv`. (Note: despite `.webp`/`.svg` being *preferred* formats per the original integration brief, no working extractor exists for them yet — see "Honest limitations" below. Extending `IMAGE_EXTENSIONS`/`image_extractor.py` to cover them is future work, not implemented today.)

### Error responses (all return a plain, human-readable JSON `detail` string — never a raw stack trace)

| Status | When | Example `detail` |
|---|---|---|
| `422` | Empty file body | `"The uploaded file is empty."` |
| `422` | Image bytes don't decode | `"That doesn't look like a valid image file -- it may be corrupted."` |
| `422` | Image too small | `"That image is too small to read reliably. Try a higher-resolution chart image."` |
| `422` | No plot border found | `"ChartSpeak couldn't find a plot area in this image. Try a clearer chart image with visible axes and gridlines."` |
| `422` | Fewer than 2 numeric Y-axis ticks OCR'd | `"ChartSpeak couldn't read enough numeric Y-axis labels on this chart to calibrate it. Try a clearer chart image with visible axis numbers."` |
| `422` | Y-axis ticks OCR'd but not spread out enough to calibrate | `"The Y-axis labels ChartSpeak found were not usably spread out on this chart."` |
| `422` | No X-axis tick labels OCR'd | `"ChartSpeak couldn't read the X-axis labels on this chart. Try a clearer chart image with visible axis labels."` |
| `422` | No colored (non-grayscale) series pixels found | `"ChartSpeak couldn't find a plotted line or set of data points on this chart. Try a chart where the data series has a distinct color from the axes and gridlines."` |
| `422` | Colors found but none line up with X-axis ticks | `"ChartSpeak found colors on this chart but couldn't line them up with the X-axis labels. Try a chart where each data point sits directly above its axis label."` |
| `422` | `.json` isn't valid JSON | `"That .json file isn't valid JSON."` |
| `422` | `.json`/`.csv` doesn't match the `ChartCSR` schema | `"That file doesn't match the expected chart format: <pydantic validation error>"` |
| `422` | Parsed chart has zero data points in every series | `"ChartSpeak couldn't find any data points in this chart."` |
| `422` | File isn't valid UTF-8 text (for `.json`/`.csv`) | `"The file isn't readable text -- it may be corrupted."` |
| `415` | Recognized-but-unsupported image type (pdf/webp/svg/gif/bmp) | `"ChartSpeak can read PNG and JPG chart images directly. PDF isn't supported yet -- ..."` |
| `415` | Any other unrecognized extension | `"Unsupported file type. Please upload a PNG/JPG chart image, a .json (ChartCSR), or a .csv (Chart4Blind export) file."` |
| `500` | Analysis pipeline itself throws unexpectedly | `"Analysis failed: <reason>"` |

Frontend code should show `detail` directly to the user (it is already written to be human-readable) and log the full response/status to the console for debugging — never surface a raw traceback.

## CORS

Configured via `CHARTSPEAK_CORS_ORIGINS` (env var), read at server startup:
- Unset or `"*"` (the default) → allows any origin. Fine for local hackathon development; **do not ship this to production**.
- A comma-separated list (e.g. `CHARTSPEAK_CORS_ORIGINS=https://chartspeak.app,https://staging.chartspeak.app`) → restricts to exactly those origins.

## How image extraction actually works (so you know what to expect)

There is no pre-existing automatic image→data pipeline anywhere in this
project to reuse (Chart4Blind's own README says its "automatic" mode
needs an external "LineFormer equivalent backend" that was never built;
this engine's own `src/schema/csr.py` is a placeholder with a
hand-typed fake sample, not real extraction code). `image_extractor.py`
is a new, from-scratch, **deterministic** digitizer — explicitly not an
LLM, and it never invents a value:

1. Find the plot's black axis-border rectangle.
2. OCR the image with Tesseract to read numeric Y-axis tick labels and
   X-axis tick labels, and fit a robust (outlier-tolerant) pixel→value
   calibration line from the Y ticks.
3. Isolate the plotted series' colored pixels (anything clearly more
   saturated than the grayscale gridlines/axes/text) and cluster them
   by **hue** (robust to anti-aliasing/JPEG-compression color noise,
   unlike matching exact RGB values). Classify each hue's connected
   components as marker-like (small, roughly square — e.g. diamond/dot
   markers) or line-stroke-like; marker centroids are preferred as the
   authoritative value at each X tick (an intentional, precise mark),
   falling back to sampling the line's pixel column when no marker is
   present there.
4. Every returned point carries a genuine confidence score derived from
   measured pixel alignment quality — not a fixed/fake number.

If any step can't find what it needs, it raises a specific
`ExtractionError` (see the table above) instead of returning a guess.

### Honest limitations

This handles clean, computer-generated line charts with visible
gridlines and printed, OCR-legible axis tick numbers/labels — e.g. an
Excel/matplotlib export or a clean screenshot. It is **not**:
- a general photograph-of-a-whiteboard/handwriting digitizer,
- a bar-chart or pie-chart digitizer (line charts only, today),
- reliable on charts with more than two data-series colors (marker↔line
  color pairing beyond a single pair is a best-effort nearest-match,
  documented as unverified),
- able to read a rotated Y-axis title (left blank — OCR on 90°-rotated
  text is unreliable, this is a known gap, not a crash).

Verified end-to-end against three real, independently-produced test
images (a bundled sample PNG, a freshly generated matplotlib PNG, and a
re-compressed JPG of the same chart) — see `tests/test_image_extractor.py`.

## Schemas

### `ChartCSR` (request body for `/analyze`; also what `.json` uploads must match)

```json
{
  "chart_id": "revenue",
  "title": "Weekly Revenue",
  "chart_type": "line",
  "x_axis": {"label": "Day", "unit": null, "values": ["Mon", "Tues", "Wed", "Thurs", "Fri"]},
  "y_axis": {"label": "Revenue", "unit": "USD", "min": null, "max": null},
  "series": [
    {
      "id": "series_0",
      "name": "Value",
      "points": [
        {"x": "Mon", "y": 301.9, "confidence": 0.975},
        {"x": "Tues", "y": 453.0, "confidence": 0.994}
      ]
    }
  ]
}
```

### `ChartSpeakResponse` (response body for both `/analyze` and `/analyze-file`)

Top level:
```json
{
  "chart_id": "revenue",
  "analysis": { "...": "AnalysisResult -- see below" },
  "salience_summaries": [ { "...": "one SalienceSummary per series" } ],
  "narrations": [ { "...": "one NarrationResult per series" } ]
}
```

`analysis` (`AnalysisResult`) — one `SeriesAnalysis` per series, plus an optional two-series `comparison`:
```json
{
  "chart_id": "revenue",
  "series": [
    {
      "series_id": "series_0",
      "name": "Value",
      "overall_trend": {"direction": "increasing", "slope": 64.84, "strength": 0.60},
      "range": {"minimum": {"x": "Wed", "y": 198.85}, "maximum": {"x": "Tues", "y": 453.79}},
      "changes": [{"from": "Mon", "to": "Tues", "absolute_change": 150.9, "percentage_change": 50.0}],
      "local_extrema": [{"type": "peak", "x": "Tues", "y": 453.79, "index": 1}],
      "turning_points": [{"type": "peak", "x": "Tues", "y": 453.79, "index": 1, "from_direction": "increasing", "to_direction": "decreasing"}],
      "sudden_changes": [{"from": "Tues", "to": "Wed", "absolute_change": -254.94, "percentage_change": -56.18, "robust_z_score": -5.77, "direction": "decrease"}],
      "outliers": [],
      "trend_breaks": []
    }
  ],
  "comparison": null
}
```

`salience_summaries[i]` (`SalienceSummary`) — ranked/scored candidate events for one series:
```json
{
  "chart_id": "revenue",
  "series_id": "series_0",
  "headline_events": [
    {"type": "valley", "x": "Wed", "y": 198.85, "index": 2, "factors": {"magnitude": 0.8, "unusualness": 0.9, "trend_disruption": 0.7, "position_importance": 0.5, "confidence": 0.99}, "salience": 0.81, "level": "high", "reason": "..."}
  ],
  "extended_events": ["... up to 7"],
  "all_events": ["... every candidate, unfiltered"],
  "counts_by_level": {"critical": 0, "high": 1, "medium": 1, "low": 1, "negligible": 0}
}
```

`narrations[i]` (`NarrationResult`) — plain-language text ready to display or speak:
```json
{
  "chart_id": "revenue",
  "series_id": "series_0",
  "text": "This is 'revenue', a line chart showing Value against Day from Mon to Fri. Overall, Value is increasing, following a moderate pattern. The most notable points are: At Wed, Value reached its lowest notable point, 198.85. Between Tues and Wed, Value fell by 254.94 (56.18%). At Tues, Value reached its highest notable point, 453.79.",
  "source": "template",
  "grounded": true
}
```
`source` is `"template"` (pure string formatting, zero LLM involvement) unless `ANTHROPIC_API_KEY` is set, in which case an LLM phrases the same whitelisted facts and `grounded` reports whether narrator.py's grounding check confirmed every number traces back to real analysis output.

## Example request (what the frontend actually sends)

```js
const formData = new FormData();
formData.append("file", file); // a real File object from an <input type="file"> or a drop event

const response = await fetch(`${API_BASE_URL}/analyze-file`, {
  method: "POST",
  body: formData, // do NOT set Content-Type -- fetch sets the multipart boundary itself
});

if (!response.ok) {
  const { detail } = await response.json().catch(() => ({ detail: "Something went wrong." }));
  throw new Error(detail);
}

const result = await response.json(); // a ChartSpeakResponse
```

## Verified example (real request/response, captured from an actual run)

```
$ curl -X POST http://127.0.0.1:8123/analyze-file -F "file=@revenue.jpg;type=image/jpeg"
HTTP/1.1 200 OK
```
```json
{
  "chart_id": "revenue",
  "analysis": {
    "chart_id": "revenue",
    "series": [{
      "series_id": "series_0",
      "name": "Value",
      "overall_trend": {"direction": "increasing", "slope": 64.8445, "strength": 0.6012},
      "range": {"minimum": {"x": "Wed", "y": 198.853}, "maximum": {"x": "Tues", "y": 453.79}},
      "changes": [
        {"from": "Mon", "to": "Tues", "absolute_change": 151.94, "percentage_change": 50.66},
        {"from": "Tues", "to": "Wed", "absolute_change": -254.94, "percentage_change": -56.18},
        {"from": "Wed", "to": "Thurs", "absolute_change": 201.75, "percentage_change": 101.46},
        {"from": "Thurs", "to": "Fri", "absolute_change": 248.87, "percentage_change": 62.06}
      ],
      "local_extrema": [
        {"type": "peak", "x": "Tues", "y": 453.79, "index": 1},
        {"type": "valley", "x": "Wed", "y": 198.853, "index": 2}
      ],
      "turning_points": ["... same two points, with from_direction/to_direction"],
      "sudden_changes": [{"from": "Tues", "to": "Wed", "absolute_change": -254.94, "percentage_change": -56.18, "robust_z_score": -5.77, "direction": "decrease"}],
      "outliers": [],
      "trend_breaks": []
    }],
    "comparison": null
  },
  "salience_summaries": ["..."],
  "narrations": [{
    "chart_id": "revenue", "series_id": "series_0",
    "text": "This is 'revenue', a line chart showing Value against Day from Mon to Fri. Overall, Value is increasing, following a moderate pattern. The most notable points are: At Wed, Value reached its lowest notable point, 198.853. Between Tues and Wed, Value fell by 254.94 (56.18%). At Tues, Value reached its highest notable point, 453.79.",
    "source": "template", "grounded": true
  }]
}
```
(True underlying values for this test chart were Mon=300, Tues=450, Wed=200, Thurs=400, Fri=650 — the extracted values are within normal OCR/pixel-measurement error of the real chart, not fabricated.)
