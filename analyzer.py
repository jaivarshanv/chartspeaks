"""
analyzer.py
-----------
Deterministic mathematical analysis.

This file takes a validated ChartCSR (from test_data.py today, from
Person 1's real pipeline later) and computes, using plain math -- no AI,
no guessing:

STAGE 1 (per series):
1. Overall trend (direction, slope, strength)
2. Minimum and maximum (with their locations)
3. Point-to-point absolute changes
4. Point-to-point percentage changes

STAGE 2 (per series): local peaks and valleys
STAGE 3 (per series): turning points (direction reversals, plateau-safe)
STAGE 4 (per series): sudden changes (statistically unusual jumps, GLOBAL)
STAGE 5 (per series): outliers (statistically unusual raw values, GLOBAL)
STAGE 6 (per series): trend breaks (statistically unusual jumps, LOCAL)

STAGE 7 (across series):
7. Crossings -- where two series swap which one is on top
8. Relative growth -- which series grew more, start to finish
9. Convergence / divergence -- are the two series getting closer or
   further apart, especially after a crossing

WHY DETERMINISTIC MATH FIRST?
Because this is "Brain A" in the architecture. Every fact ChartSpeak ever
speaks, writes, or vibrates to a user must trace back to a real
calculation done here. Nothing here is allowed to be invented later by
an LLM.
"""

from typing import List, Optional
import statistics
import numpy as np
from scipy import stats

from models import (
    ChartCSR,
    Series,
    Trend,
    ExtremePoint,
    RangeInfo,
    PointChange,
    LocalExtremum,
    TurningPoint,
    SuddenChange,
    Outlier,
    TrendBreak,
    SeriesAnalysis,
    Crossing,
    SeriesGrowth,
    GrowthComparison,
    ConvergenceInfo,
    MultiSeriesComparison,
    AnalysisResult,
)

# The "0.6745" constant and "3.5" threshold below are a well-known
# combination from robust-statistics literature (Iglewicz & Hoya, 1993)
# for turning MAD into something comparable to a normal z-score. We are
# using them as a reasonable, documented STARTING POINT -- not as a
# scientifically validated setting for chart data specifically. Section
# 19 of the project spec is explicit about this: treat these as tunable,
# and we'll revisit them once we've tested against more real charts.
MODIFIED_Z_SCALING_CONSTANT = 0.6745
SUDDEN_CHANGE_THRESHOLD = 3.5

# A separate constant (even though it currently has the same value as
# SUDDEN_CHANGE_THRESHOLD) because "how unusual is this jump?" and "how
# unusual is this raw value?" are different questions that may need
# different sensitivity once we tune against real charts.
OUTLIER_Z_THRESHOLD = 3.5

# How many of the immediately preceding changes we look at to decide
# "what has this chart been doing lately." Too small a window (like 1)
# would be too easily thrown off by ordinary noise; too large a window
# starts to blur into "the whole chart's behavior," which is exactly
# the global comparison find_sudden_changes() already does. 3 is a
# reasonable, tunable starting point for a hackathon prototype.
TREND_BREAK_WINDOW_SIZE = 3
TREND_BREAK_Z_THRESHOLD = 3.5

# How small a step has to be before we treat it as "flat" instead of a
# real increase/decrease. Real extracted chart data (from Person 1's
# pipeline) can have tiny floating-point wobble -- e.g. a value that
# "should" be exactly 20.0 might come out as 19.9999997. Without this
# threshold, that kind of noise would look like a direction change and
# create a fake turning point. Our fake test data is exact integers, so
# this threshold won't change today's output -- it's here so the same
# code keeps working once real, noisier data arrives.
FLAT_EPSILON = 1e-9


def calculate_trend(y_values: List[float]) -> Trend:
    """
    Fits a straight line through the points using LINEAR REGRESSION and
    reads off two things from that line:

    - slope: how much y changes, on average, per step along x.
      Positive slope = line tilts up overall. Negative = tilts down.

    - strength: the absolute value of the CORRELATION COEFFICIENT (r).
      r ranges from -1 to 1 and measures how well a single straight line
      actually explains the data. r close to 1 or -1 means "the points
      really do sit close to a straight line." r close to 0 means "a
      straight line is a poor description," even if the average
      direction is technically up or down. We take abs(r) so strength
      is always reported on a simple 0 (weak) to 1 (strong) scale.

    We use scipy.stats.linregress, a standard, well-tested function
    that computes exactly this in one call.
    """
    x_indices = np.arange(len(y_values))  # [0, 1, 2, 3, 4, 5] -- treats each point as one evenly spaced step
    result = stats.linregress(x_indices, y_values)
    slope = float(result.slope)
    r_value = float(result.rvalue)

    # A small "dead zone" around zero avoids calling a nearly-flat line
    # "increasing" or "decreasing" just because of tiny rounding noise.
    if slope > 0.01:
        direction = "increasing"
    elif slope < -0.01:
        direction = "decreasing"
    else:
        direction = "flat"

    return Trend(
        direction=direction,
        slope=round(slope, 4),
        strength=round(abs(r_value), 4),
    )


def find_range(x_values: List[str], y_values: List[float]) -> RangeInfo:
    """
    Finds the global minimum and maximum values, and WHERE (which x label)
    they occurred.

    np.argmin / np.argmax return the INDEX of the smallest/largest value,
    not the value itself -- that's why we use that index to look up both
    the x label and the y value.
    """
    min_index = int(np.argmin(y_values))
    max_index = int(np.argmax(y_values))

    minimum = ExtremePoint(x=x_values[min_index], y=y_values[min_index])
    maximum = ExtremePoint(x=x_values[max_index], y=y_values[max_index])

    return RangeInfo(minimum=minimum, maximum=maximum)


def calculate_changes(x_values: List[str], y_values: List[float]) -> List[PointChange]:
    """
    Walks through the series one step at a time and records, for every
    consecutive pair of points:

    - absolute_change: to_y - from_y (a plain difference, in the chart's units)
    - percentage_change: how big that difference is RELATIVE to the
      starting value, as a percentage: (absolute_change / from_y) * 100

    Special case: if from_y is 0, percentage change is mathematically
    undefined (division by zero), so we report 0.0 and note it below.
    A future stage can decide to special-case this more carefully.
    """
    changes: List[PointChange] = []

    for i in range(len(y_values) - 1):
        from_x, to_x = x_values[i], x_values[i + 1]
        from_y, to_y = y_values[i], y_values[i + 1]

        absolute_change = to_y - from_y

        if from_y == 0:
            percentage_change = 0.0  # avoids ZeroDivisionError; not mathematically meaningful
        else:
            percentage_change = (absolute_change / from_y) * 100.0

        changes.append(
            PointChange(
                from_x=from_x,
                to_x=to_x,
                absolute_change=round(absolute_change, 2),
                percentage_change=round(percentage_change, 2),
            )
        )

    return changes


def find_local_extrema(x_values: List[str], y_values: List[float]) -> List[LocalExtremum]:
    """
    STAGE 2: Local peaks and valleys.

    A LOCAL PEAK is a point that is strictly higher than both its
    immediate left and right neighbors:      previous < current > next
    A LOCAL VALLEY is a point that is strictly lower than both:
                                              previous > current < next

    We can only check points that HAVE two neighbors, so the very first
    and very last point in the series are skipped -- they only have one
    neighbor each, so "local peak/valley" isn't a meaningful question for
    them (the global min/max check in find_range() already covers
    endpoints being the overall highest/lowest value).

    IMPORTANT LIMITATION (by design, for Stage 2):
    We use STRICT inequalities (< and >), not <= or >=. This means a
    "plateau" -- e.g. 10, 20, 20, 20, 10 -- will NOT be flagged, because
    the middle points are equal to their neighbor, not strictly greater.
    Stage 3's find_turning_points() below handles plateaus correctly --
    this function is kept simple on purpose, as a straightforward,
    textbook building block.
    """
    extrema: List[LocalExtremum] = []

    for i in range(1, len(y_values) - 1):
        previous_y = y_values[i - 1]
        current_y = y_values[i]
        next_y = y_values[i + 1]

        if previous_y < current_y > next_y:
            extrema.append(LocalExtremum(type="peak", x=x_values[i], y=current_y, index=i))
        elif previous_y > current_y < next_y:
            extrema.append(LocalExtremum(type="valley", x=x_values[i], y=current_y, index=i))

    return extrema


def _sign_of(diff: float) -> int:
    """
    Turns a raw difference into a direction: +1 (increasing),
    -1 (decreasing), or 0 (flat). Anything smaller than FLAT_EPSILON in
    magnitude counts as flat, so tiny floating-point noise can never be
    mistaken for a real direction change.
    """
    if abs(diff) < FLAT_EPSILON:
        return 0
    return 1 if diff > 0 else -1


def find_turning_points(x_values: List[str], y_values: List[float]) -> List[TurningPoint]:
    """
    STAGE 3: Turning points -- places where the series changes direction.

    THE APPROACH:
    1. Compute the sign of every step (+1 up, -1 down, 0 flat).
    2. Walk through those signs left to right, remembering the last
       direction we actually saw that WASN'T flat.
    3. Whenever we hit a new non-flat direction that's the OPPOSITE of
       the last non-flat direction we remember, that's a turning point.
       Flat steps in between are simply skipped over while looking --
       they don't reset or confuse the comparison.

    WHY THIS HANDLES PLATEAUS CORRECTLY:
    Example: 10, 20, 20, 20, 10
    Steps:   +10,  0,  0, -10
    Signs:    +1,  0,  0,  -1
    The two zeros are skipped. We compare the +1 directly against the
    -1 that eventually follows, and correctly report ONE turning point
    (at the last "20" before the drop) instead of three confusing ones
    or none at all.

    WHY THIS MATCHES local_extrema FOR OUR TEST CHART:
    When there are no flat stretches, a "sign flip" and a "previous 
    current > next" local peak/valley describe the exact same point.
    You'll see turning_points come out identical to Stage 2's
    local_extrema for our May-spike chart -- that's expected, and a
    good sign both methods are correct. Turning-point detection is the
    more general of the two because of how it handles plateaus.
    """
    diffs = [y_values[i + 1] - y_values[i] for i in range(len(y_values) - 1)]
    signs = [_sign_of(diff) for diff in diffs]

    turning_points: List[TurningPoint] = []
    last_nonzero_sign = None  # the last real (non-flat) direction we've seen so far

    for i, sign in enumerate(signs):
        if sign == 0:
            continue  # flat step: skip it, don't update anything, just keep looking

        if last_nonzero_sign is not None and sign != last_nonzero_sign:
            # Direction flipped. The shared point is index i (the point
            # this diff STARTS from) -- for a plain up-then-down run
            # that's the exact peak; for a plateau, it's the last point
            # of the flat stretch, right before the reversal.
            if last_nonzero_sign == 1 and sign == -1:
                point_type = "peak"
                from_direction, to_direction = "increasing", "decreasing"
            else:
                point_type = "valley"
                from_direction, to_direction = "decreasing", "increasing"

            turning_points.append(
                TurningPoint(
                    type=point_type,
                    x=x_values[i],
                    y=y_values[i],
                    index=i,
                    from_direction=from_direction,
                    to_direction=to_direction,
                )
            )

        last_nonzero_sign = sign

    return turning_points


def _modified_z_scores(values: List[float]) -> List[float]:
    """
    Shared helper: computes a modified z-score (median/MAD based) for
    every value in the list, in the same order. Both find_sudden_changes
    (Stage 4, scores the CHANGES) and find_outliers (Stage 5, scores the
    RAW VALUES) call this -- the statistical method is identical, only
    what list gets passed in differs. See find_sudden_changes()'s
    docstring for the full explanation of median/MAD and why we use them
    instead of mean/standard-deviation.
    """
    if len(values) == 0:
        return []

    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)

    scores: List[float] = []
    for value, deviation in zip(values, deviations):
        if mad > 0:
            z_score = MODIFIED_Z_SCALING_CONSTANT * (value - median) / mad
        else:
            # No typical spread to divide by: anything different from
            # the median is maximally unusual, anything equal to it is
            # perfectly normal. This avoids a ZeroDivisionError.
            z_score = 0.0 if deviation == 0 else float("inf")
            if value < median:
                z_score = -z_score
        scores.append(z_score)

    return scores


def find_sudden_changes(changes: List[PointChange]) -> List[SuddenChange]:
    """
    STAGE 4: Distinguish normal movement from statistically UNUSUAL movement.
    This is a GLOBAL comparison: every change is scored against the
    median/MAD of every OTHER change in the whole chart. See
    find_trend_breaks() below for the LOCAL version of this same idea.

    THE PROBLEM WITH A FIXED THRESHOLD ("anything above 10 is sudden"):
    A change of 10 might be huge for a chart that normally moves by 1,
    and completely unremarkable for a chart that normally swings by 50.
    "Sudden" only makes sense relative to how much THIS chart's values
    normally move -- so we compare each change against the DISTRIBUTION
    of all the changes in this same chart.

    WHY MEDIAN + MAD INSTEAD OF MEAN + STANDARD DEVIATION?
    The classic way to ask "how unusual is this number?" is a z-score:
    (value - mean) / standard_deviation. The problem is that both the
    mean and the standard deviation are themselves dragged around by
    extreme values. If a chart already has one dramatic outlier (like
    our May -> June drop of -35), that single point inflates the
    standard deviation so much that it can make OTHER real outliers look
    "normal" by comparison -- statisticians call this "masking."

    The MEDIAN (the middle value when everything is sorted) barely moves
    when one value is extreme. The MEDIAN ABSOLUTE DEVIATION (MAD) --
    the median of how far every point sits from that median -- inherits
    the same resistance to outliers. That's why MAD-based detection is
    the standard recommendation for exactly this scenario (a chart that
    may already contain the outlier you're trying to detect).

    THE FORMULA (a "modified z-score"), computed by _modified_z_scores():
        modified_z = 0.6745 * (value - median) / MAD
    The constant 0.6745 rescales MAD so that, for data that roughly
    follows a normal ("bell curve") distribution, this number means
    roughly the same thing as an ordinary z-score. A common rule of
    thumb (which we use here) is to flag anything with
    |modified_z| > 3.5 as unusual.
    """
    if len(changes) == 0:
        return []

    absolute_changes = [c.absolute_change for c in changes]
    z_scores = _modified_z_scores(absolute_changes)

    sudden_changes: List[SuddenChange] = []

    for change, z_score in zip(changes, z_scores):
        if abs(z_score) > SUDDEN_CHANGE_THRESHOLD:
            direction = "increase" if change.absolute_change > 0 else "decrease"
            rounded_z = round(z_score, 4) if z_score not in (float("inf"), float("-inf")) else z_score
            sudden_changes.append(
                SuddenChange(
                    from_x=change.from_x,
                    to_x=change.to_x,
                    absolute_change=change.absolute_change,
                    percentage_change=change.percentage_change,
                    robust_z_score=rounded_z,
                    direction=direction,
                )
            )

    return sudden_changes


def find_outliers(x_values: List[str], y_values: List[float]) -> List[Outlier]:
    """
    STAGE 5: Robust outlier detection on the RAW VALUES themselves.

    This answers a different question than find_sudden_changes() above:
    "is this y-value, by itself, unlike the other y-values in this
    chart?" -- as opposed to "is the JUMP into or out of this point
    unusual?" The two are independent. A point can be an outlier without
    its neighboring changes being sudden, and vice versa.

    KEY PRINCIPLE FROM THE SPEC: A global maximum is NOT automatically
    an outlier. For example, in the series 10, 20, 30, 40, 50, the value
    50 is the largest number, but it fits the pattern perfectly -- it's
    exactly where you'd expect the next step to land. Statistically it
    is completely unsurprising. Compare that to 10, 11, 10, 12, 50, 11:
    here 50 is wildly outside how this series normally behaves, even
    though the RAW NUMBER 50 is identical in both examples. "High" and
    "statistically unusual" are different concepts, and this function
    only flags the second one -- using the exact same median/MAD
    modified z-score method as find_sudden_changes(), just applied to
    y_values instead of the changes between them.

    A NOTE ON SAMPLE SIZE: MAD-based detection gets more statistically
    meaningful as you feed it more points. Our 6-point test chart is a
    small sample, so treat its outlier results as illustrative rather
    than definitive -- this matters more once real charts (usually with
    many more points) start flowing through this function.
    """
    if len(y_values) == 0:
        return []

    z_scores = _modified_z_scores(y_values)

    outliers: List[Outlier] = []
    for index, (x, y, z_score) in enumerate(zip(x_values, y_values, z_scores)):
        if abs(z_score) > OUTLIER_Z_THRESHOLD:
            rounded_z = round(z_score, 4) if z_score not in (float("inf"), float("-inf")) else z_score
            outliers.append(Outlier(x=x, y=y, index=index, robust_z_score=rounded_z))

    return outliers


def _robust_z_against_window(value: float, window_values: List[float]) -> float:
    """
    Like _modified_z_scores(), but scores a single VALUE against a
    separate reference window instead of scoring every item of one list
    against itself. This is the key difference between Stage 4/5 (global:
    "compare this item to everything in the chart") and Stage 6 (local:
    "compare this item to only what happened right before it").
    """
    median = statistics.median(window_values)
    deviations = [abs(v - median) for v in window_values]
    mad = statistics.median(deviations)

    if mad > 0:
        return MODIFIED_Z_SCALING_CONSTANT * (value - median) / mad

    if value == median:
        return 0.0
    return float("-inf") if value < median else float("inf")


def find_trend_breaks(
    changes: List[PointChange],
    window_size: int = TREND_BREAK_WINDOW_SIZE,
) -> List[TrendBreak]:
    """
    STAGE 6: Trend breaks -- local behavior that no longer matches the
    pattern the chart had recently established.

    WHY THIS IS DIFFERENT FROM find_sudden_changes() (STAGE 4):
    Stage 4 asks "is this change unusual compared to EVERY change in the
    whole chart?" That's a GLOBAL comparison. This function asks "is
    this change unusual compared only to the handful of changes right
    before it?" That's a LOCAL comparison, and the two can disagree.

    A concrete case where they disagree: imagine a chart that opens with
    wild swings and then settles into a calm, steady pattern. Those
    early wild swings make the WHOLE CHART's "typical spread" (used by
    Stage 4) large -- so a moderate disruption to the later calm section
    might not look unusual next to the earlier chaos, even though it is
    a real break from what had become the local norm. Because this
    function only looks at the window_size steps immediately before each
    point, it isn't fooled by chaos that happened somewhere else in the
    chart.

    HOW IT WORKS:
    For every change starting at index `window_size` (we need at least
    that many earlier changes to know what "the recent pattern" even is),
    compare it against the median/MAD of the window_size changes
    immediately before it using the same modified-z-score idea as
    Stages 4 and 5.

    LIMITATION: with very short series (like our 6-point test chart),
    the "local window" and "the whole chart" overlap almost completely,
    so trend breaks and sudden changes will often agree. The distinction
    matters more -- and becomes genuinely necessary -- on longer, more
    realistic charts, which is what Person 1's real data will look like.

    A SECOND, HONEST LIMITATION: a small window (like 3) can produce a
    MAD of exactly 0 just by coincidence -- e.g. a window like
    [50, -50, 50] has a repeated value, which drags MAD to 0 even though
    the window is clearly not "calm." When that happens, this function
    treats ANY different value as maximally unusual (an infinite
    z-score), which can over-flag during genuinely chaotic stretches of
    a chart. This is a known trade-off of MAD with very small samples --
    worth revisiting (e.g. a minimum-window-size rule, or blending in
    another spread measure) once we test against real, longer charts.
    """
    trend_breaks: List[TrendBreak] = []

    for i in range(window_size, len(changes)):
        window = changes[i - window_size : i]
        window_values = [c.absolute_change for c in window]
        current = changes[i]

        z_score = _robust_z_against_window(current.absolute_change, window_values)

        if abs(z_score) > TREND_BREAK_Z_THRESHOLD:
            rounded_z = round(z_score, 4) if z_score not in (float("inf"), float("-inf")) else z_score
            local_median = round(statistics.median(window_values), 2)
            reason = (
                f"This change ({current.absolute_change}) breaks from the pattern of "
                f"the preceding {window_size} steps (typical recent change was about {local_median})."
            )
            trend_breaks.append(
                TrendBreak(
                    from_x=current.from_x,
                    to_x=current.to_x,
                    absolute_change=current.absolute_change,
                    local_robust_z_score=rounded_z,
                    reason=reason,
                )
            )

    return trend_breaks


def analyze_series(series: Series) -> SeriesAnalysis:
    """
    Runs every Stage 1-6 calculation on ONE series and bundles the
    results into a SeriesAnalysis. This is exactly the body that used
    to live directly inside analyze_chart() before Stage 7 -- pulling it
    out into its own function is what lets analyze_chart() below call it
    once per series instead of being hardcoded to csr.series[0].
    """
    x_values = [point.x for point in series.points]
    y_values = [point.y for point in series.points]

    trend = calculate_trend(y_values)
    range_info = find_range(x_values, y_values)
    changes = calculate_changes(x_values, y_values)
    local_extrema = find_local_extrema(x_values, y_values)
    turning_points = find_turning_points(x_values, y_values)
    sudden_changes = find_sudden_changes(changes)
    outliers = find_outliers(x_values, y_values)
    trend_breaks = find_trend_breaks(changes)

    return SeriesAnalysis(
        series_id=series.id,
        name=series.name,
        overall_trend=trend,
        range=range_info,
        changes=changes,
        local_extrema=local_extrema,
        turning_points=turning_points,
        sudden_changes=sudden_changes,
        outliers=outliers,
        trend_breaks=trend_breaks,
    )


def find_crossings(
    x_values: List[str],
    series_a_values: List[float],
    series_b_values: List[float],
    series_a_id: str,
    series_b_id: str,
) -> List[Crossing]:
    """
    STAGE 7: Finds every point where two series swap which one is higher.

    THE APPROACH (deliberately similar to find_turning_points in Stage 3):
    1. At every shared x position, compute (series_a - series_b).
    2. Take the SIGN of that difference: +1 means A is on top, -1 means
       B is on top, 0 means they're exactly tied at that point.
    3. Walk through the signs left to right, remembering the last
       non-tied leader. Whenever the leader changes, that's a crossing.
       Exact ties are skipped over rather than treated as crossings by
       themselves -- a crossing is a CHANGE of leadership, not merely a
       tie (the two series could touch and then A could stay on top).

    We reuse _sign_of() from Stage 3 for the "is this basically zero"
    check, for the same reason: real extracted data can have tiny
    floating-point noise that shouldn't be mistaken for an exact tie.
    """
    diffs = [a - b for a, b in zip(series_a_values, series_b_values)]
    signs = [_sign_of(diff) for diff in diffs]

    crossings: List[Crossing] = []
    last_nonzero_sign = None
    last_nonzero_index = None

    for i, sign in enumerate(signs):
        if sign == 0:
            continue  # an exact tie at this point; skip, keep looking

        if last_nonzero_sign is not None and sign != last_nonzero_sign:
            leading_before = series_a_id if last_nonzero_sign == 1 else series_b_id
            leading_after = series_a_id if sign == 1 else series_b_id

            crossings.append(
                Crossing(
                    from_x=x_values[last_nonzero_index],
                    to_x=x_values[i],
                    from_index=last_nonzero_index,
                    to_index=i,
                    leading_before=leading_before,
                    leading_after=leading_after,
                )
            )

        last_nonzero_sign = sign
        last_nonzero_index = i

    return crossings


def _percentage_change(start: float, end: float) -> float:
    """First-to-last percentage change, guarding the same divide-by-zero case as calculate_changes()."""
    if start == 0:
        return 0.0
    return round(((end - start) / start) * 100.0, 2)


def compare_two_series(
    x_values: List[str],
    series_a: Series,
    series_b: Series,
) -> MultiSeriesComparison:
    """
    STAGE 7: Everything about how two series relate to each other.

    This function assumes both series share the same x positions (the
    same months, categories, etc.) -- which is the normal case for two
    lines plotted on the same chart. It computes three things:

    1. CROSSINGS -- via find_crossings() above.

    2. RELATIVE GROWTH -- each series' percentage change from its first
       point to its last, and which one grew more (using the SIGNED
       percentage change, so "grew more" correctly means "increased
       more," not just "changed by a bigger number either direction").

    3. CONVERGENCE / DIVERGENCE -- whether the GAP between the two
       series (the absolute difference at each point) is shrinking
       (converging) or growing (diverging). We reuse calculate_trend()
       for this -- it doesn't care whether the numbers it's given are
       chart values or gap sizes, it just fits a line and reports the
       direction and strength.

       We specifically look at the gap ONLY AFTER THE LAST CROSSING when
       one exists, because that's the period your spec's own example
       asks about ("do they diverge AFTER crossing?"). Looking at the
       whole chart instead can hide this: two series that start far
       apart, converge, cross, and then diverge by the same amount will
       average out to "no trend" if you look at the whole gap sequence
       at once, even though the after-crossing behavior is a clear,
       real divergence.
    """
    series_a_values = [p.y for p in series_a.points]
    series_b_values = [p.y for p in series_b.points]

    crossings = find_crossings(x_values, series_a_values, series_b_values, series_a.id, series_b.id)

    growth_a = SeriesGrowth(
        series_id=series_a.id,
        direction=calculate_trend(series_a_values).direction,
        percentage_change=_percentage_change(series_a_values[0], series_a_values[-1]),
    )
    growth_b = SeriesGrowth(
        series_id=series_b.id,
        direction=calculate_trend(series_b_values).direction,
        percentage_change=_percentage_change(series_b_values[0], series_b_values[-1]),
    )
    greater_growth_series_id = (
        series_a.id if growth_a.percentage_change >= growth_b.percentage_change else series_b.id
    )
    growth_comparison = GrowthComparison(
        series_a=growth_a,
        series_b=growth_b,
        greater_growth_series_id=greater_growth_series_id,
    )

    gap_values = [abs(a - b) for a, b in zip(series_a_values, series_b_values)]

    if crossings:
        start_index = crossings[-1].to_index
        segment_gap_values = gap_values[start_index:]
        segment = "after_last_crossing"
    else:
        segment_gap_values = gap_values
        segment = "whole_series"

    if len(segment_gap_values) < 2:
        # Not enough points left after the last crossing to fit a trend
        # line -- be honest about that instead of reporting a made-up answer.
        convergence = ConvergenceInfo(direction="unknown", strength=0.0, segment="insufficient_data")
    else:
        gap_trend = calculate_trend(segment_gap_values)
        if gap_trend.direction == "increasing":
            convergence_direction = "diverging"  # gap is growing
        elif gap_trend.direction == "decreasing":
            convergence_direction = "converging"  # gap is shrinking
        else:
            convergence_direction = "stable"
        convergence = ConvergenceInfo(
            direction=convergence_direction,
            strength=gap_trend.strength,
            segment=segment,
        )

    return MultiSeriesComparison(
        series_a_id=series_a.id,
        series_b_id=series_b.id,
        crossings=crossings,
        growth_comparison=growth_comparison,
        convergence=convergence,
    )


def analyze_chart(csr: ChartCSR) -> AnalysisResult:
    """
    The main entry point.

    Analyzes EVERY series in the chart (not just csr.series[0] anymore --
    that hardcoding is gone as of Stage 7). If the chart has exactly two
    series, it also builds a MultiSeriesComparison between them.

    WHY ONLY EXACTLY TWO SERIES FOR COMPARISON (for now):
    Your spec's own examples (Series A vs Series B) are always pairwise.
    Comparing three or more series well (which pairs to compare? every
    combination? just consecutive ones?) is a real design question we
    haven't answered yet, and answering it before two-series comparison
    is solid would be building on a shaky foundation. So for now: 1
    series -> comparison is None. 2 series -> one full comparison. 3+
    series -> comparison is still None, and this is a known, documented
    scope limit for a later stage, not an oversight.
    """
    series_analyses = [analyze_series(series) for series in csr.series]

    comparison: Optional[MultiSeriesComparison] = None
    if len(csr.series) == 2:
        x_values = [point.x for point in csr.series[0].points]
        comparison = compare_two_series(x_values, csr.series[0], csr.series[1])

    return AnalysisResult(
        chart_id=csr.chart_id,
        series=series_analyses,
        comparison=comparison,
    )