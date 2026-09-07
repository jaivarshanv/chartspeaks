"""
image_extractor.py
-------------------
STAGE 14 (real image -> ChartCSR): a deterministic, non-LLM chart digitizer.

WHY THIS FILE EXISTS
Neither this engine (Person 2's analysis/salience/narration code) nor the
team's Chart4Blind app (Person 1's code, a separate repo) contains a
working, automatic image-to-data pipeline:

  - This engine's own `src/schema/csr.py` is a schema definition plus a
    hand-typed FAKE sample CSR ("This is NOT a real extraction yet --
    it's fake/sample data" -- see tests/test_csr_sample.py). No image
    ever touches it.
  - Chart4Blind's "automatic" mode calls a remote `/api/trans4line/`
    endpoint that its own README says requires "a LineFormer equivalent
    backend" that "must be running" -- and isn't. Chart4Blind's real,
    working mode is a human manually dragging calibration points and
    clicking data points onto the image in a browser editor.

So there is no existing extraction implementation to reuse for a fully
automatic "JPG in, ChartCSR out" flow. Rather than fabricate values with
an LLM (explicitly forbidden -- see architectural principle in the brief:
"The LLM must never invent chart values"), this module does real,
measurable computer vision:

  1. Find the plot's axis border (the black rectangle around the data).
  2. OCR the image (Tesseract -- a real, deterministic, offline OCR
     engine, the same one Chart4Blind's own OCRFeatures.tsx uses
     client-side) to read the Y-axis tick numbers and X-axis tick
     labels, and calibrate pixel-space to real chart units from them.
  3. Find the plotted series by isolating saturated (colorful) pixels
     from the grayscale gridlines/text/axes, and trace each series
     color's vertical position at every X tick column.

Every number this module returns is either read directly off the image
by OCR or measured from actual pixel positions -- never guessed. If a
step can't find what it needs (no axis border, no numeric Y ticks, no
colored series), it raises ExtractionError with a specific, honest
reason instead of returning made-up data.

SCOPE / HONEST LIMITATIONS (see API_CONTRACT.md):
This handles clean, computer-generated line charts with visible grid
lines and printed axis tick labels -- e.g. an Excel/matplotlib export,
a clean screenshot, a scanned textbook figure. It is not a general
photograph-of-a-whiteboard digitizer, and it does not (yet) attempt bar
charts. That's the same class of input the rest of ChartSpeak's error
copy already anticipates: "Try a clearer chart image with visible axes
and labels."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytesseract

from models import ChartCSR, ChartPoint, Series, XAxis, YAxis

# Temporary debugging logs (see api.py) -- stage-by-stage progress
# through the digitizer, metadata only, never image pixel data.
logger = logging.getLogger("chartspeak")


class ExtractionError(Exception):
    """Raised when the image genuinely can't be digitized -- always
    carries a specific, user-showable reason (never a stack trace)."""


# ---------------------------------------------------------------------
# Small geometry / OCR helpers
# ---------------------------------------------------------------------

@dataclass
class Token:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


_NUMBER_RE = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")


def _parse_number(text: str) -> Optional[float]:
    cleaned = text.strip().replace(",", "").replace("%", "")
    if not cleaned or not _NUMBER_RE.match(text.strip()):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _ocr_tokens(gray: np.ndarray, upscale: int = 2) -> list[Token]:
    """Run Tesseract on an upscaled copy (small chart images OCR far
    more reliably at 2x-3x) and map boxes back to original pixel space."""
    big = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT, config="--psm 11")
    tokens: list[Token] = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        if int(data.get("conf", ["0"])[i] or -1) < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append(Token(
            text=text,
            x0=x // upscale, y0=y // upscale,
            x1=(x + w) // upscale, y1=(y + h) // upscale,
        ))
    return tokens


def _find_plot_rect(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the axis border: the largest near-black rectangular
    contour that plausibly frames the plotted data. Returns
    (left, top, right, bottom) in pixel coordinates."""
    h, w = gray.shape
    dark = (gray < 80).astype(np.uint8) * 255
    # Close small gaps in the border (e.g. anti-aliased corners) before
    # looking for its outline.
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        # A plot border should span a real fraction of the image and
        # not be a sliver (a single gridline) or the whole image edge.
        if cw < w * 0.25 or ch < h * 0.15:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + cw, y + ch)
    if best is None:
        raise ExtractionError(
            "ChartSpeak couldn't find a plot area in this image. "
            "Try a clearer chart image with visible axes and gridlines."
        )
    return best


def _robust_linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Fit y = slope*x + intercept while tolerating a few bad points --
    OCR sometimes misreads one tick label (e.g. a tick mark's dash
    fusing into the digits and turning "25" into "254") without
    misreading the others. A plain least-squares fit lets one such
    outlier drag the whole calibration off; this is a small RANSAC:
    try every pair of ticks as a candidate line, count how many of the
    OTHER ticks that line actually predicts well, and keep the
    best-supported line (refit by least squares over its full inlier
    set). Falls back to a plain fit when there are too few points for
    voting to mean anything."""
    n = len(xs)
    if n <= 2:
        return tuple(np.polyfit(xs, ys, 1))
    value_scale = max(float(np.ptp(ys)), 1.0) * 0.06
    best_inliers: Optional[np.ndarray] = None
    best_spread = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if xs[i] == xs[j]:
                continue
            slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
            intercept = ys[i] - slope * xs[i]
            residual = np.abs(slope * xs + intercept - ys)
            inliers = np.where(residual <= value_scale)[0]
            spread = abs(float(xs[j]) - float(xs[i]))
            # More inliers wins outright. On a tie (common with only a
            # handful of ticks, where a wrong pair can "explain" just
            # itself as well as a correct pair does), prefer the pair
            # spanning more pixels: axis ticks are evenly spaced in
            # real charts, so the two extreme ticks anchor a much more
            # numerically stable line than two closely-spaced ones, and
            # OCR misreads happen more often on ticks that cross
            # gridlines in the middle of the axis than on the end ticks.
            better = best_inliers is None or len(inliers) > len(best_inliers) or (
                len(inliers) == len(best_inliers) and spread > best_spread
            )
            if better:
                best_inliers, best_spread = inliers, spread
    if best_inliers is None or len(best_inliers) < 2:
        return tuple(np.polyfit(xs, ys, 1))
    return tuple(np.polyfit(xs[best_inliers], ys[best_inliers], 1))


def _cluster_rows(values: list[float], gap: float) -> list[list[int]]:
    """Cluster indices of `values` into groups whose values are within
    `gap` of their neighbours once sorted -- used to split OCR tokens
    below the plot into an "axis tick" row vs an "axis title" row."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for idx in order:
        if groups and values[idx] - values[groups[-1][-1]] <= gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


# ---------------------------------------------------------------------
# Series (plotted line / marker) detection
#
# Naive "match this exact RGB" tracing turned out to badly fragment a
# single anti-aliased line into several near-duplicate "colors" (edge
# pixels blend toward white/black, shifting R/G/B by a lot while barely
# shifting HUE). So instead: work in HSV, cluster by HUE only (robust to
# anti-aliasing), then classify each hue cluster's connected components
# as either "markers" (several small, roughly square blobs -- e.g. the
# diamond dots on a line-with-markers chart) or "line strokes" (long,
# thin, elongated blobs). Markers are the most trustworthy signal for a
# point's real value (they mark an exact spot the chart's author placed
# on purpose), so they're preferred whenever present; a plain line's
# hue cluster is sampled column-by-column as a fallback.
# ---------------------------------------------------------------------

@dataclass
class ColorCluster:
    hue: int
    mask: np.ndarray  # boolean, shape = plot region (top-left = plot_box's top-left)
    components: list[dict]
    is_marker_like: bool


def _hue_distance(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 180 - d)


def _cluster_hues(hues: np.ndarray, merge_dist: int = 12, min_pixels: int = 6) -> list[int]:
    """Greedy clustering of a circular hue histogram (OpenCV hue range
    is 0-179, i.e. degrees/2 around a full color wheel) into
    representative hues, folding in any hue within `merge_dist` degrees
    of an existing cluster.

    IMPORTANT: hue is CIRCULAR (179 sits right next to 0 -- both are
    "red"). A plain arithmetic mean of two merged hues breaks exactly
    there: averaging hue 0 and hue 178 the naive way gives ~89 (a
    greenish hue that is nowhere near either red sample), not ~179. On
    a real red chart line, JPEG compression alone scatters red pixels'
    measured hue across both sides of that 0/179 seam, so without a
    circular-aware mean the very first couple of merges corrupt the
    cluster's representative hue and the rest of red's pixels stop
    matching it -- fragmenting one real line into several bogus
    "series". We track each cluster's running mean as a 2D unit-vector
    sum (the standard way to average angles) and only convert back to
    a 0-179 hue value when a representative is actually needed."""
    values, counts = np.unique(hues, return_counts=True)
    order = np.argsort(-counts)
    reps: list[int] = []
    vector_sums: list[tuple[float, float]] = []  # (sum of cos, sum of sin), weighted by pixel count
    for i in order:
        hue, count = int(values[i]), int(counts[i])
        if count < min_pixels:
            continue
        angle = hue * (2 * np.pi / 180.0)
        for ci, rep in enumerate(reps):
            if _hue_distance(hue, rep) <= merge_dist:
                cos_sum, sin_sum = vector_sums[ci]
                cos_sum += np.cos(angle) * count
                sin_sum += np.sin(angle) * count
                vector_sums[ci] = (cos_sum, sin_sum)
                mean_angle = np.arctan2(sin_sum, cos_sum)
                if mean_angle < 0:
                    mean_angle += 2 * np.pi
                reps[ci] = int(round(mean_angle * 180.0 / (2 * np.pi))) % 180
                break
        else:
            reps.append(hue)
            vector_sums.append((np.cos(angle) * count, np.sin(angle) * count))
    return reps


def _analyze_color_group(hue_map: np.ndarray, sat_mask: np.ndarray, target_hue: int, merge_dist: int, marker_max_size: float) -> Optional[ColorCluster]:
    dist = np.minimum(np.abs(hue_map - target_hue), 180 - np.abs(hue_map - target_hue))
    group_mask = (dist <= merge_dist) & sat_mask
    if group_mask.sum() < 6:
        return None
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(group_mask.astype(np.uint8) * 255, connectivity=8)
    components = []
    marker_like_count = 0
    for i in range(1, n):
        x, y, w, hgt, area = stats[i]
        if area < 3:
            continue
        components.append({"bbox": (int(x), int(y), int(w), int(hgt)), "area": int(area), "centroid": (float(centroids[i][0]), float(centroids[i][1]))})
        if w < marker_max_size and hgt < marker_max_size:
            marker_like_count += 1
    if not components:
        return None
    return ColorCluster(hue=target_hue, mask=group_mask, components=components, is_marker_like=marker_like_count == len(components))


def _sample_mask_column(mask: np.ndarray, x_pixel: int, search_radius: int = 4) -> Optional[tuple[float, int]]:
    """Mean pixel-y (and match count) of a boolean mask at a column,
    searching a small horizontal radius if the exact column is empty
    (e.g. it falls in the gap between two line segments)."""
    height, width = mask.shape
    for radius in range(search_radius + 1):
        for dx in ([0] if radius == 0 else [-radius, radius]):
            xi = x_pixel + dx
            if xi < 0 or xi >= width:
                continue
            ys = np.where(mask[:, xi])[0]
            if len(ys):
                return float(ys.mean()), len(ys)
    return None


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def extract_csr(image_bytes: bytes, chart_id: str) -> ChartCSR:
    """The one function api.py calls. Turns raw image bytes into a
    verified ChartCSR, or raises ExtractionError with a specific,
    user-showable reason."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ExtractionError("That doesn't look like a valid image file -- it may be corrupted.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    logger.info("extraction started: chart_id=%r image=%dx%d", chart_id, w, h)
    if h < 50 or w < 50:
        raise ExtractionError("That image is too small to read reliably. Try a higher-resolution chart image.")

    plot_box = _find_plot_rect(gray)
    left, top, right, bottom = plot_box
    logger.info("plot area found: box=%s", plot_box)

    logger.info("OCR started (tesseract)")
    tokens = _ocr_tokens(gray)
    logger.info("OCR complete: %d text tokens found", len(tokens))

    # --- Y axis: numeric tick labels to the left of the plot area ---
    y_ticks: list[tuple[float, float]] = []  # (pixel_y, real_value)
    for t in tokens:
        if t.cx >= left - 2:
            continue
        value = _parse_number(t.text)
        if value is not None:
            y_ticks.append((t.cy, value))
    y_ticks = sorted(set(y_ticks))
    if len(y_ticks) < 2:
        raise ExtractionError(
            "ChartSpeak couldn't read enough numeric Y-axis labels on this chart to calibrate it. "
            "Try a clearer chart image with visible axis numbers."
        )
    py = np.array([p for p, _ in y_ticks])
    vals = np.array([v for _, v in y_ticks])
    if np.ptp(py) < 1e-6:
        raise ExtractionError("The Y-axis labels ChartSpeak found were not usably spread out on this chart.")
    slope, intercept = _robust_linear_fit(py, vals)

    def pixel_y_to_value(y_pixel: float) -> float:
        return float(slope * y_pixel + intercept)

    # --- X axis: tick labels below the plot area ---
    below = [t for t in tokens if t.cy > bottom + 1 and t.x0 < right + 15 and t.x1 > left - 15]
    if not below:
        raise ExtractionError(
            "ChartSpeak couldn't read the X-axis labels on this chart. "
            "Try a clearer chart image with visible axis labels."
        )
    rows = _cluster_rows([t.cy for t in below], gap=8)
    # The tick-label row is whichever row sits closest to the plot's
    # bottom edge and has more than one label (an axis title like "Day"
    # is usually alone on its own row further down).
    rows.sort(key=lambda idxs: min(below[i].cy for i in idxs))
    tick_row = next((idxs for idxs in rows if len(idxs) > 1), rows[0])
    x_tokens = sorted((below[i] for i in tick_row), key=lambda t: t.cx)
    x_labels = [(t.cx, t.text) for t in x_tokens]

    # --- Chart title: text above the plot area, if any ---
    above = [t for t in tokens if t.cy < top - 1]
    title = " ".join(t.text for t in sorted(above, key=lambda t: (t.cy, t.cx))) if above else chart_id

    # --- X axis label: a lone row below the tick row (e.g. "Day") ---
    x_axis_label = ""
    if len(rows) > 1:
        title_row = [idxs for idxs in rows if idxs is not tick_row]
        if title_row:
            x_axis_label = " ".join(below[i].text for i in sorted(title_row[-1], key=lambda i: below[i].cx))

    # --- Y axis label: rotated text far to the left of the Y ticks ---
    # (best-effort only -- OCR on 90-degree-rotated text is unreliable,
    # so an empty y-axis label is expected and fine, not an error.)
    y_axis_label = ""

    # --- Series colors: cluster by hue, classify markers vs line strokes ---
    region_bgr = bgr[top:bottom, left:right]
    region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    rb, gb, bb = region_bgr[:, :, 2].astype(int), region_bgr[:, :, 1].astype(int), region_bgr[:, :, 0].astype(int)
    sat_range = np.maximum(np.maximum(rb, gb), bb) - np.minimum(np.minimum(rb, gb), bb)
    sat_mask = sat_range > 30
    if sat_mask.sum() < 6:
        raise ExtractionError(
            "ChartSpeak couldn't find a plotted line or set of data points on this chart. "
            "Try a chart where the data series has a distinct color from the axes and gridlines."
        )
    hue_map = region_hsv[:, :, 0].astype(int)
    dominant_hues = _cluster_hues(hue_map[sat_mask])
    if not dominant_hues:
        raise ExtractionError(
            "ChartSpeak couldn't find a plotted line or set of data points on this chart. "
            "Try a chart where the data series has a distinct color from the axes and gridlines."
        )

    x_positions = [x for x, _ in x_labels]
    tick_gap = (max(x_positions) - min(x_positions)) / max(1, len(x_positions) - 1) if len(x_positions) > 1 else float(right - left)
    marker_max_size = max(6.0, tick_gap * 0.6)

    clusters = [c for c in (_analyze_color_group(hue_map, sat_mask, hue, 12, marker_max_size) for hue in dominant_hues) if c is not None]
    if not clusters:
        raise ExtractionError(
            "ChartSpeak couldn't find a plotted line or set of data points on this chart. "
            "Try a chart where the data series has a distinct color from the axes and gridlines."
        )

    marker_clusters = [c for c in clusters if c.is_marker_like and len(c.components) >= 2]
    line_clusters = [c for c in clusters if not (c.is_marker_like and len(c.components) >= 2)]
    used_line_indices: set[int] = set()

    def points_for_marker_cluster(marker: ColorCluster, fallback_line: Optional[ColorCluster]) -> list[ChartPoint]:
        points: list[ChartPoint] = []
        max_match_dist = tick_gap * 0.6
        for x_pixel, label in x_labels:
            # Component centroids and the line mask are in region-local
            # coordinates (origin at plot_box's top-left); x_labels are
            # in full-image coordinates -- convert before comparing.
            x_local = x_pixel - left
            best_comp, best_dist = None, max_match_dist
            for comp in marker.components:
                d = abs(comp["centroid"][0] - x_local)
                if d < best_dist:
                    best_dist, best_comp = d, comp
            if best_comp is not None:
                y_pixel = best_comp["centroid"][1] + top
                # A genuine, measured confidence: how precisely this
                # marker's center lines up with the tick's x position.
                confidence = round(max(0.5, 1.0 - best_dist / max_match_dist), 3)
            elif fallback_line is not None:
                hit = _sample_mask_column(fallback_line.mask, int(round(x_local)))
                if hit is None:
                    continue
                y_pixel = hit[0] + top
                confidence = round(min(0.6, hit[1] / 3.0), 3)
            else:
                continue
            points.append(ChartPoint(x=label, y=round(pixel_y_to_value(y_pixel), 4), confidence=confidence))
        return points

    def points_for_line_cluster(line: ColorCluster) -> list[ChartPoint]:
        points: list[ChartPoint] = []
        for x_pixel, label in x_labels:
            x_local = x_pixel - left
            hit = _sample_mask_column(line.mask, int(round(x_local)))
            if hit is None:
                continue
            y_pixel, pixel_count = hit
            # A genuine, measured confidence: how much of this column's
            # thickness actually matched the line's color (capped at 1).
            confidence = round(min(1.0, pixel_count / 3.0), 3)
            points.append(ChartPoint(x=label, y=round(pixel_y_to_value(y_pixel + top), 4), confidence=confidence))
        return points

    series_list: list[Series] = []
    for marker in marker_clusters:
        # A line-with-markers chart pairs one marker color with one line
        # color (e.g. blue diamonds on a red line). With more than one
        # marker color, pairing is a best-effort nearest-available match
        # rather than a verified correspondence -- documented as a known
        # limitation for genuinely multi-series marker charts.
        fallback = next((line for i, line in enumerate(line_clusters) if i not in used_line_indices), None)
        points = points_for_marker_cluster(marker, fallback)
        if fallback is not None and points:
            used_line_indices.add(line_clusters.index(fallback))
        if len(points) >= 2:
            series_list.append(Series(id=f"series_{len(series_list)}", name=f"Series {len(series_list) + 1}", points=points))

    for i, line in enumerate(line_clusters):
        if i in used_line_indices:
            continue
        points = points_for_line_cluster(line)
        if len(points) >= 2:
            series_list.append(Series(id=f"series_{len(series_list)}", name=f"Series {len(series_list) + 1}", points=points))

    if not series_list:
        raise ExtractionError(
            "ChartSpeak found colors on this chart but couldn't line them up with the X-axis labels. "
            "Try a chart where each data point sits directly above its axis label."
        )
    if len(series_list) == 1:
        series_list[0] = Series(id=series_list[0].id, name="Value", points=series_list[0].points)

    logger.info(
        "series detection complete: %d series, %s",
        len(series_list), [(s.name, len(s.points)) for s in series_list],
    )

    return ChartCSR(
        chart_id=chart_id,
        title=title.strip() or chart_id,
        chart_type="line",
        x_axis=XAxis(label=x_axis_label.strip(), unit=None, values=[lbl for _, lbl in x_labels]),
        y_axis=YAxis(label=y_axis_label.strip(), unit=None, min=None, max=None),
        series=series_list,
    )
