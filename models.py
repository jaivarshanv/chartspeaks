"""
models.py
---------
This file defines the "shape" of our data using Pydantic.

WHAT IS PYDANTIC?
Pydantic lets us describe data as Python classes (based on BaseModel).
When we feed it raw data (like a dictionary that came from JSON),
it automatically CHECKS that the data matches the shape we described,
and CONVERTS types where possible (e.g. "80" -> 80.0).
If the data is wrong (missing field, wrong type), Pydantic raises a
clear error immediately instead of letting a bad value silently break
our math later. This is called "validation".

This file has two groups of models:
1. INPUT models -> describe the CSR (Chart Semantic Representation)
   that Person 1's Chart4Blind pipeline will eventually produce.
2. OUTPUT models -> describe the AnalysisResult that OUR engine produces.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# =========================================================
# INPUT MODELS: THE CSR (Chart Semantic Representation)
# =========================================================

class ChartPoint(BaseModel):
    """One single data point on a chart, e.g. { "x": "May", "y": 95, "confidence": 0.94 }"""
    x: str
    y: float
    confidence: float = 1.0  # if Person 1 doesn't provide one yet, assume perfect confidence


class XAxis(BaseModel):
    """Describes the horizontal axis of the chart."""
    label: str
    unit: Optional[str] = None
    values: List[str]


class YAxis(BaseModel):
    """Describes the vertical axis of the chart."""
    label: str
    unit: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


class Series(BaseModel):
    """One line/bar/set of points on the chart. A chart can have multiple series (Stage 7)."""
    id: str
    name: str
    points: List[ChartPoint]


class ChartCSR(BaseModel):
    """
    The full Chart Semantic Representation.
    This is the CONTRACT between Person 1 (Chart4Blind/extraction) and
    Person 2 (this analysis engine). As long as whoever builds the CSR
    (fake data today, Person 1's real pipeline later) produces something
    matching this shape, nothing in analyzer.py has to change.
    """
    chart_id: str
    title: str
    chart_type: str
    x_axis: XAxis
    y_axis: YAxis
    series: List[Series]


# =========================================================
# OUTPUT MODELS: THE ANALYSIS RESULT
# =========================================================

class Trend(BaseModel):
    """
    Describes the overall direction of the chart.
    direction: "increasing" | "decreasing" | "flat"
    slope: how steeply the line rises/falls per step, on average
           (positive = rising, negative = falling)
    strength: 0.0-1.0, how well a single straight line actually
              describes the data (this comes from a correlation
              coefficient). A low strength means the chart does NOT
              behave like a clean straight line even if the overall
              direction is technically up or down.
    """
    direction: str
    slope: float
    strength: float


class ExtremePoint(BaseModel):
    """A single labeled point, used for the minimum and maximum of the chart."""
    x: str
    y: float


class RangeInfo(BaseModel):
    """The lowest and highest points found in the series."""
    minimum: ExtremePoint
    maximum: ExtremePoint


class PointChange(BaseModel):
    """
    The change between two consecutive points.
    We use "from_x"/"to_x" as the Python field names because "from" is a
    reserved word in Python (you can't name a variable "from"). But when we
    print this as JSON, we tell Pydantic to use the aliases "from" and "to"
    instead, so the JSON output still matches the schema in the spec.
    """
    model_config = ConfigDict(populate_by_name=True)

    from_x: str = Field(alias="from")
    to_x: str = Field(alias="to")
    absolute_change: float
    percentage_change: float


class LocalExtremum(BaseModel):
    """
    A LOCAL peak or valley: a point that is higher (or lower) than the
    points immediately next to it. This is different from the global
    minimum/maximum in RangeInfo -- a chart can have several local peaks
    and valleys, but only one global minimum and one global maximum.

    type: "peak" | "valley"
    index: the point's position in the series (0 = first point). We keep
           this because later stages (turning points, trend breaks,
           salience) need to know how far apart events are, not just
           their labels.
    """
    type: str
    x: str
    y: float
    index: int


class TurningPoint(BaseModel):
    """
    A point where the DIRECTION of the series reverses -- it was going up
    and starts going down, or vice versa.

    This is found differently than LocalExtremum: instead of comparing a
    point to its immediate neighbors, we track the SIGN of each step
    (increasing / decreasing / flat) across the whole series and look for
    where that sign flips. This makes it possible to correctly skip over
    "flat" stretches (a plateau of equal values) without inventing a fake
    reversal in the middle of them -- see find_turning_points() for how.

    type: "peak" (was increasing, now decreasing) or
          "valley" (was decreasing, now increasing)
    from_direction / to_direction: the direction before and after this
          point, spelled out for clarity ("increasing" / "decreasing")
    """
    type: str
    x: str
    y: float
    index: int
    from_direction: str
    to_direction: str


class SuddenChange(BaseModel):
    """
    One point-to-point change that is STATISTICALLY UNUSUAL compared to
    the other changes in this same chart -- not just "a big number" in
    isolation.

    robust_z_score: how many "typical spreads" away from the median
        change this one is, using the median absolute deviation (MAD)
        instead of a plain average/standard deviation. See
        find_sudden_changes() in analyzer.py for the full explanation of
        why we use this instead of a simple std-dev z-score.
    direction: "increase" | "decrease" -- which way the sudden change went.
    """
    model_config = ConfigDict(populate_by_name=True)

    from_x: str = Field(alias="from")
    to_x: str = Field(alias="to")
    absolute_change: float
    percentage_change: float
    robust_z_score: float
    direction: str


class Outlier(BaseModel):
    """
    A single RAW VALUE (not a change between two points) that is
    statistically unusual compared to every other value in the series.

    This is deliberately a different question than SuddenChange: an
    outlier looks at "how strange is this y-value among all the
    y-values," while a sudden change looks at "how strange is this jump
    among all the jumps." A point can be one, both, or neither -- see
    find_outliers() in analyzer.py for a worked example of a chart's
    maximum value NOT being an outlier.
    """
    x: str
    y: float
    index: int
    robust_z_score: float


class TrendBreak(BaseModel):
    """
    A change that violates the LOCALLY ESTABLISHED pattern of the steps
    right before it -- a different question than SuddenChange, which
    compares a jump against every change in the WHOLE chart.

    Why keep these separate? A chart can have a lot of volatility early
    on (which makes the whole-chart "typical spread" large) and then
    settle into a very calm, consistent pattern. A moderate disruption
    to that calm pattern might not be extreme enough to stand out
    against the noisy early section, but it is still a real break from
    what had become locally normal. find_trend_breaks() in analyzer.py
    has a worked example of exactly this situation.

    local_robust_z_score: how unusual this change is compared only to
        the window_size changes immediately before it (not the whole chart).
    reason: a short, human-readable explanation of what was compared to what.
    """
    model_config = ConfigDict(populate_by_name=True)

    from_x: str = Field(alias="from")
    to_x: str = Field(alias="to")
    absolute_change: float
    local_robust_z_score: float
    reason: str


class SeriesAnalysis(BaseModel):
    """
    Everything we know about ONE series after mathematical analysis.

    STAGE 7 CHANGE: through Stage 6, this exact set of fields WAS
    AnalysisResult itself, sitting directly at the top level, because we
    only ever analyzed csr.series[0]. Now that we support multiple
    series, we need a container that can hold ONE OF THESE PER SERIES --
    so this class is the old AnalysisResult body, renamed and nested
    under a list. See AnalysisResult below for the new top-level shape.
    """
    series_id: str
    name: str
    overall_trend: Trend
    range: RangeInfo
    changes: List[PointChange]
    local_extrema: List[LocalExtremum]
    turning_points: List[TurningPoint]
    sudden_changes: List[SuddenChange]
    outliers: List[Outlier]
    trend_breaks: List[TrendBreak]


class Crossing(BaseModel):
    """
    A point where two series swap which one is higher.

    Like PointChange, this describes an INTERVAL (from one x label to
    the next), because the crossing usually happens somewhere between
    two plotted points rather than exactly on one of them, and because
    the two series might briefly sit at the exact same value at one of
    our points (a "tie") without leadership actually swapping -- see
    find_crossings() in analyzer.py.

    leading_before / leading_after: the series_id that was on top right
        before, and right after, this crossing.
    """
    model_config = ConfigDict(populate_by_name=True)

    from_x: str = Field(alias="from")
    to_x: str = Field(alias="to")
    from_index: int
    to_index: int
    leading_before: str
    leading_after: str


class SeriesGrowth(BaseModel):
    """
    One series' overall change from its first point to its last,
    restated here (alongside the comparison) so Person 3's narration
    layer doesn't have to go cross-reference the separate `series` list
    just to say "Series A increased by 400%."
    """
    series_id: str
    direction: str  # copied from that series' overall_trend.direction
    percentage_change: float  # first point to last point, not step-by-step


class ConvergenceInfo(BaseModel):
    """
    Whether the GAP between the two series (|series_a - series_b| at
    each shared point) is growing (diverging), shrinking (converging),
    or not moving in a clear direction (stable), over some segment of
    the chart.

    segment: which points were used to decide this --
        "after_last_crossing" (the most informative answer, when a
        crossing exists and there's enough data after it), or
        "whole_series" (used when there's no crossing, or too few
        points remain after the last one to say anything meaningful).
    strength: reuses the same correlation-based 0-1 confidence idea as
        Trend.strength in calculate_trend().
    """
    direction: str  # "converging" | "diverging" | "stable"
    strength: float
    segment: str


class GrowthComparison(BaseModel):
    """Side-by-side growth of the two series, plus which one grew more."""
    series_a: SeriesGrowth
    series_b: SeriesGrowth
    greater_growth_series_id: str


class MultiSeriesComparison(BaseModel):
    """
    Everything about how two series relate to EACH OTHER -- as opposed
    to SeriesAnalysis, which only ever looks at one series in isolation.
    Only produced when a chart has exactly two series (see analyze_chart()
    in analyzer.py for why we're not yet handling three or more).
    """
    series_a_id: str
    series_b_id: str
    crossings: List[Crossing]
    growth_comparison: GrowthComparison
    convergence: ConvergenceInfo


class SalienceFactors(BaseModel):
    """
    The five 0.0-1.0 inputs that get combined into one salience score.
    See salience.py for exactly how each one is computed and why these
    particular weights were chosen as a first, documented guess rather
    than a scientifically validated setting.
    """
    magnitude: float
    unusualness: float
    trend_disruption: float
    position_importance: float
    confidence: float


class CandidateEvent(BaseModel):
    """
    One scored, ranked candidate event -- either a POINT event (peak or
    valley: x/y are set, from_x/to_x are None) or an INTERVAL event
    (increase or decrease: from_x/to_x/absolute_change are set, x/y are
    None).

    We use ONE flexible model with optional fields, rather than two
    separate types, so the whole ranked list -- points and intervals
    mixed together -- can live in a single flat list and be sorted by
    `salience` directly, without needing separate lists merged later.
    Check `type` to know which fields are populated:
        "peak" / "valley"       -> x, y are set
        "increase" / "decrease" -> from_x, to_x, absolute_change are set
    """
    model_config = ConfigDict(populate_by_name=True)

    type: str
    x: Optional[str] = None
    y: Optional[float] = None
    from_x: Optional[str] = Field(default=None, alias="from")
    to_x: Optional[str] = Field(default=None, alias="to")
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None
    index: int
    factors: SalienceFactors
    salience: float
    level: str
    reason: str


class SalienceSummary(BaseModel):
    """
    STAGE 9: the ranked-AND-filtered result for one series -- what
    Person 3's narration layer actually consumes by default, plus the
    full data still available "on request" (spec section 17's two
    layers: a meaningful summary by default, exact values on demand).

    headline_events: the short list (at most 3) meant for the DEFAULT
        narration -- "here's what matters most about this chart."
    extended_events: a slightly longer list (at most 7), still
        excluding "negligible" events, for a user who wants a bit more
        detail without going all the way to raw numbers.
    all_events: every candidate score_series_salience() computed, with
        nothing removed -- this IS the "give me the exact numbers" layer.
    counts_by_level: how many candidates fell into each salience band,
        always including all five level names even when a count is 0,
        so Person 3's UI can read every key without checking first.
    """
    chart_id: str
    series_id: str
    headline_events: List[CandidateEvent]
    extended_events: List[CandidateEvent]
    all_events: List[CandidateEvent]
    counts_by_level: Dict[str, int]


class AnalysisResult(BaseModel):
    """
    STAGE 7 CHANGE -- new top-level shape.

    Through Stage 6, this class WAS what SeriesAnalysis is now: a flat
    bag of chart_id + series_id + all the per-series metrics, because we
    only ever analyzed one series. That flat shape literally cannot
    represent "two series and how they relate," so this stage
    restructures it:

    - `series` now holds ONE SeriesAnalysis per series in the chart.
      For a single-series chart (like our original Jan-Jun example),
      this is just a list with one entry -- every number you verified
      in Stages 1-6 is still there, just nested one level deeper.
    - `comparison` is populated only when the chart has exactly two
      series, and holds everything about how they relate to each other
      (crossings, relative growth, convergence/divergence). It's `None`
      for single-series charts.
    """
    chart_id: str
    series: List[SeriesAnalysis]
    comparison: Optional[MultiSeriesComparison] = None