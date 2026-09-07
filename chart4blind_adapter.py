"""
chart4blind_adapter.py
-----------------------
STAGE 12: turning Person 1's real Chart4Blind output into a ChartCSR.

WHAT I FOUND BY ACTUALLY READING PERSON 1'S CODE (not guessing):

Chart4Blind is a React/TypeScript app (Chart4Blind-main), backed by a
Python OCR/CV service. The pipeline is: an uploaded chart image goes
through an OCR backend endpoint (/api/trans4line/) that returns a DENSE
pixel-by-pixel trace of each detected line -- hundreds of {x, y} pairs
per line, one for roughly every horizontal pixel, in raw PIXEL space.
That is NOT the same thing as chart "data points": it's a traced curve,
not a small set of meaningful values. The app then lets the user (via a
slider, in AIFeatures.tsx) or an automated step reduce that dense trace
down to a much smaller set of actual DataPoints per line, and lets the
user place/drag two calibration points on each axis (X1/X2, Y1/Y2) plus
a scale type (linear / logarithmic / time). Only AFTER that calibration
is applied (scaleValueToCalibration() in dotInteractionUtility.ts) do
the pixel coordinates turn into real chart values -- exactly what
happens today when a user clicks "Export as CSV" in ExportModal.tsx.

THE INTEGRATION DECISION THIS FILE MAKES:
I deliberately do NOT reimplement scaleValueToCalibration() in Python.
That calibration math (including log-scale and time-scale handling, and
a line-intersection trick to find the pixel-space axis origin) already
exists, is already used, and would just be a second copy to keep in
sync if duplicated here -- exactly the mistake pipeline.py was written
to avoid in Stage 11. Instead, this adapter's input boundary is
Chart4Blind's EXISTING CSV export format (getCSVData() in
ExportModal.tsx) -- calibration has already happened by the time that
CSV exists, and it requires ZERO code changes on Person 1's side to
start using today: any chart already digitized in the app can be
exported as CSV right now and fed straight into this function.

WHAT THIS ADAPTER CANNOT KNOW FROM A CSV ALONE (documented, not hidden):
- Per-point OCR confidence: Chart4Blind doesn't track this at all, for
  any point. Every ChartPoint gets the schema's own default (1.0) --
  the same "if Person 1 doesn't give us a confidence value, assume
  perfect confidence" default from Stage 1, not a new assumption.
- Whether the x-axis was originally a date/log scale: the CSV only
  contains the final calibrated NUMBER (e.g. a raw Unix timestamp, if
  the chart used a time scale) -- the scale type itself isn't in the
  export. See _format_x_label() below for how this is handled honestly.
- y_axis min/max: not part of the CSV, left as None (ChartCSR already
  supports that as optional).
"""

import re
from typing import List, Optional

from models import ChartCSR, XAxis, YAxis, Series, ChartPoint


def _format_x_label(value: float) -> str:
    """
    Turns a calibrated numeric x-value from the CSV into a display label.

    WHY THIS MATTERS: analyzer.py never parses ChartPoint.x as a number
    (verified -- it only uses x as an ordered label; trend/slope math
    works on evenly-spaced step indices, not the label's numeric value).
    So converting a float to a string here is always safe for Stages
    1-11 -- but it's still worth doing carefully, because raw floating
    point math can produce ugly noise like "3.0000000000000004" for
    what should just be "3". We round first, then format, and let whole
    numbers print without a trailing ".0".
    """
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


def _slugify(name: str, fallback: str) -> str:
    """Turns a line's title into a simple id, e.g. 'Series A' -> 'series_a'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or fallback


def parse_chart4blind_csv(csv_text: str, chart_id: str) -> ChartCSR:
    """
    Parses the EXACT CSV format Chart4Blind's ExportModal.getCSVData()
    produces today, into a validated ChartCSR.

    The format (verified against the real getCSVData() source, including
    its exact whitespace, which is NOT symmetric across the four header
    lines -- the JS string-builds them slightly inconsistently):

        Title, <chart title>
         Description,<description>
         xAxis Label, <x axis label>
         yAxis Label, <y axis label>
        <line 1 title>
        xCoordinate, yCoordinate
        <x>, <y>
        <x>, <y>
        <line 2 title>
        xCoordinate, yCoordinate
        <x>, <y>
        ...

    Rather than relying on fixed line positions for the per-line blocks
    (fragile if a future version of the export changes formatting), each
    non-header line is classified by what it actually contains: a line
    that parses as exactly two numbers is a data point; a line that
    reads "xCoordinate, yCoordinate" (any case/spacing) is the repeated
    sub-header and is skipped; anything else is a new series' title.
    """
    lines = [line for line in csv_text.split("\n")]

    def _header_value(line: str) -> str:
        _, _, value = line.partition(",")
        return value.strip()

    title = _header_value(lines[0])
    # lines[1] is Description -- not part of ChartCSR's schema, so it's
    # read (to correctly advance past it) but intentionally discarded.
    x_label = _header_value(lines[2])
    y_label = _header_value(lines[3])

    series_list: List[Series] = []
    current_name: Optional[str] = None
    current_points: List[ChartPoint] = []
    used_ids = set()

    def _flush_current_series():
        if current_name is not None and current_points:
            base_id = _slugify(current_name, f"series_{len(series_list) + 1}")
            series_id = base_id
            suffix = 2
            while series_id in used_ids:
                series_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(series_id)
            series_list.append(Series(id=series_id, name=current_name, points=list(current_points)))

    for raw_line in lines[4:]:
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].lower() == "xcoordinate" and parts[1].lower() == "ycoordinate":
            continue  # the repeated per-series sub-header -- not data

        if len(parts) == 2:
            try:
                x_value = float(parts[0])
                y_value = float(parts[1])
            except ValueError:
                x_value = y_value = None
            if x_value is not None:
                current_points.append(ChartPoint(x=_format_x_label(x_value), y=y_value, confidence=1.0))
                continue

        # Anything that isn't a recognized sub-header or a numeric pair
        # is a new series starting -- flush whatever we were building.
        _flush_current_series()
        current_name = line
        current_points = []

    _flush_current_series()

    return ChartCSR(
        chart_id=chart_id,
        title=title,
        chart_type="line",  # Chart4Blind only supports line charts today (ChartTypes.LINECHART)
        x_axis=XAxis(label=x_label, unit=None, values=[]),
        y_axis=YAxis(label=y_label, unit=None, min=None, max=None),
        series=series_list,
    )
