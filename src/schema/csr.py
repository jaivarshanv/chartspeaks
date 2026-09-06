# csr.py
# This file defines the "Chart Semantic Representation" (CSR).
# CSR is the standard structured format that Person 1's extraction
# pipeline outputs, and that Person 2's Salience Engine consumes.
#
# We use Python's built-in "dataclasses" module. A dataclass is just
# a regular class, but Python auto-generates the boring parts for us
# (like __init__), so we only have to declare the fields.

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChartPoint:
    # A single (x, y) coordinate in REAL chart units (not pixels).
    x: float
    y: float


@dataclass
class ChartSeries:
    # One line/bar/series on the chart, made up of many ChartPoints.
    name: str
    points: List[ChartPoint] = field(default_factory=list)


@dataclass
class ChartAxis:
    # Describes one axis of the chart (the x-axis or the y-axis).
    label: str          # e.g. "Month"
    scale_type: str     # one of: "linear", "logarithmic", "time"
    unit: str           # e.g. "units", "%", "" if none
    values: List[str] = field(default_factory=list)  # axis tick labels


@dataclass
class ChartSemanticRepresentation:
    # This is the top-level CSR object. It's the ONLY thing
    # Person 2 needs to understand — nothing about pixels,
    # OCR, or how we extracted it.
    chart_type: str      # one of: "line", "bar", "scatter", "area", "unknown"
    title: str
    x_axis: ChartAxis
    y_axis: ChartAxis
    series: List[ChartSeries] = field(default_factory=list)