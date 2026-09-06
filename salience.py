"""
salience.py
-----------
STAGE 8: The Salience Engine -- ChartSpeak's first big novelty.

Stages 1-7 (analyzer.py) produce a lot of individual, TRUE, deterministic
facts about a chart: peaks, valleys, turning points, statistically
unusual changes and values, trend breaks, crossings. That's necessary,
but a real chart can easily produce a dozen or more of these facts, and
narrating every single one defeats the whole point of ChartSpeak. The
project spec's own principle: the Salience Engine answers "what deserves
attention?" -- a different question from "what exists?"

THE FUNNEL THIS FILE COMPLETES:
    100 raw points
        -> Stages 2-6 already narrowed this down to a handful of
           CANDIDATES: local extrema, turning points, outliers, sudden
           changes, trend breaks. Nothing in this file re-scans the raw
           points -- it only looks at points/changes some earlier stage
           already flagged as worth a second look.
        -> THIS FILE scores every candidate on five factors and ranks
           them, highest salience first (score_series_salience), THEN
           (as of Stage 9) filters that ranked list down to the "3-7
           things that matter" (filter_headline_events /
           filter_meaningful_events / build_salience_summary).

THE FIVE FACTORS (each normalized to roughly 0.0-1.0):
    magnitude            -- how far the value/change stands out on the
                             chart's OWN scale (its min-to-max range)
    unusualness          -- how statistically rare the value/change is,
                             reusing the z-scores Stages 4 and 5 already
                             computed (this file invents no new statistics)
    trend_disruption     -- whether this point/change breaks the
                             established direction, reusing Stage 3's
                             turning points and Stage 6's trend-break
                             z-scores
    position_importance  -- a simple recency heuristic: later points in
                             the series score slightly higher, on the
                             (debatable, tunable) idea that "what's
                             happening most recently" tends to matter
                             more in a narrated summary
    confidence           -- Person 1's OCR/extraction confidence for the
                             point(s) involved, carried through from the
                             CSR (this is the FIRST stage that actually
                             uses ChartPoint.confidence)

Final salience = a WEIGHTED SUM of those five factors.

*** IMPORTANT HONESTY NOTE (project spec section 19) ***
The weights below (0.30 / 0.25 / 0.25 / 0.10 / 0.10) are the spec's own
suggested STARTING POINT. We have not tuned or validated them against
real charts or real users. Treat every number in this file as a
reasonable, documented first guess -- not a scientifically proven
result. They are named constants specifically so they're easy to find
and change once we start testing against real feedback.
"""

from typing import Dict, List, Set
import statistics

from models import Series, SeriesAnalysis, CandidateEvent, SalienceFactors, SalienceSummary

# ---------------------------------------------------------------------
# STAGE 9: how many events survive filtering.
#
# HEADLINE_MAX_EVENTS is the short list meant for the DEFAULT narration
# (spec: "3-7 highly salient events", we start at the tight end, 3).
# EXTENDED_MAX_EVENTS is a looser "give me a bit more" list.
# Both numbers are the spec's own suggestion, not something we've
# tested with real users yet -- easy to retune here once we do.
# ---------------------------------------------------------------------
HEADLINE_MAX_EVENTS = 3
EXTENDED_MAX_EVENTS = 7
NEGLIGIBLE_LEVEL = "negligible"
ALL_SALIENCE_LEVELS = ["critical", "high", "medium", "low", "negligible"]

# ---------------------------------------------------------------------
# WEIGHTS -- must sum to 1.0 so a final score built from five factors
# that are each already in [0, 1] naturally stays in [0, 1] too.
# ---------------------------------------------------------------------
WEIGHT_MAGNITUDE = 0.30
WEIGHT_UNUSUALNESS = 0.25
WEIGHT_TREND_DISRUPTION = 0.25
WEIGHT_POSITION_IMPORTANCE = 0.10
WEIGHT_CONFIDENCE = 0.10

# How large a robust z-score (from Stages 4/5/6) counts as "maximally
# unusual" (unusualness or trend_disruption = 1.0). Our significance
# threshold for FLAGGING something as sudden/an outlier/a trend break is
# 3.5 (see analyzer.py); this reference is set well above that so a
# borderline-flagged event doesn't already max out the scale, while a
# dramatically unusual one (like the z=26 example from Stage 5) does.
# Tunable -- there is nothing sacred about "10".
Z_SCORE_SATURATION_REFERENCE = 10.0

# ---------------------------------------------------------------------
# SALIENCE LEVELS -- named bands over the final 0.0-1.0 score, exactly
# as specified in the project spec. These exist purely so a human (or
# the Stage 10 LLM) can talk about "a high-salience event" instead of a
# raw decimal. Change the thresholds here if playtesting says they feel
# wrong -- nothing else in the codebase depends on these exact numbers.
# ---------------------------------------------------------------------
def get_salience_level(score: float) -> str:
    if score < 0.25:
        return "negligible"
    if score < 0.50:
        return "low"
    if score < 0.75:
        return "medium"
    if score < 0.90:
        return "high"
    return "critical"


def _clip01(value: float) -> float:
    """
    Keeps a factor inside [0.0, 1.0]. This also safely handles the
    float('inf') z-scores that Stages 4 and 6 can produce in their
    MAD-is-zero edge case: min(1.0, inf) is simply 1.0 in Python, no
    special-casing needed.
    """
    return max(0.0, min(1.0, value))


def _combine(factors: SalienceFactors) -> float:
    """The weighted sum described in the module docstring, rounded and clipped."""
    score = (
        WEIGHT_MAGNITUDE * factors.magnitude
        + WEIGHT_UNUSUALNESS * factors.unusualness
        + WEIGHT_TREND_DISRUPTION * factors.trend_disruption
        + WEIGHT_POSITION_IMPORTANCE * factors.position_importance
        + WEIGHT_CONFIDENCE * factors.confidence
    )
    return round(_clip01(score), 4)


def _point_candidates(series: Series, analysis: SeriesAnalysis) -> List[CandidateEvent]:
    """
    Builds one CandidateEvent for every point that Stage 2 (local
    extrema), Stage 3 (turning points), or Stage 5 (outliers) already
    flagged. A point that never showed up in any of those three lists
    is not a candidate at all -- it was already judged "unremarkable"
    by earlier, purely mathematical stages, so there's nothing for the
    Salience Engine to score.
    """
    y_values = [p.y for p in series.points]
    n = len(y_values)
    median_y = statistics.median(y_values)
    value_range = analysis.range.maximum.y - analysis.range.minimum.y
    half_range = value_range / 2 if value_range > 0 else 0.0

    extrema_by_index = {e.index: e for e in analysis.local_extrema}
    outliers_by_index = {o.index: o for o in analysis.outliers}
    turning_indices: Set[int] = {t.index for t in analysis.turning_points}

    candidate_indices = set(extrema_by_index) | set(outliers_by_index) | turning_indices

    candidates: List[CandidateEvent] = []
    for index in sorted(candidate_indices):
        point = series.points[index]
        extremum = extrema_by_index.get(index)
        outlier = outliers_by_index.get(index)
        is_turning_point = index in turning_indices

        # Fall back to "above/below the median" if this point is a
        # turning point or outlier but, unusually, not a clean local
        # extremum (can happen right at a plateau boundary).
        event_type = extremum.type if extremum else ("peak" if point.y >= median_y else "valley")

        magnitude = _clip01(abs(point.y - median_y) / half_range) if half_range > 0 else 0.0
        unusualness = _clip01(abs(outlier.robust_z_score) / Z_SCORE_SATURATION_REFERENCE) if outlier else 0.0
        trend_disruption = 1.0 if is_turning_point else 0.0
        position_importance = round(index / (n - 1), 4) if n > 1 else 0.0
        confidence = point.confidence

        factors = SalienceFactors(
            magnitude=round(magnitude, 4),
            unusualness=round(unusualness, 4),
            trend_disruption=trend_disruption,
            position_importance=position_importance,
            confidence=confidence,
        )
        salience = _combine(factors)

        reasons = []
        if extremum:
            reasons.append(f"a local {extremum.type}")
        if is_turning_point:
            reasons.append("a turning point in the series' direction")
        if outlier:
            reasons.append(f"a statistical outlier (robust z-score {outlier.robust_z_score})")
        reason = (
            f"{point.x} ({point.y}) is " + ", and ".join(reasons)
            + f". It sits {round(magnitude, 2)} of the way across this chart's own value range."
        )

        candidates.append(
            CandidateEvent(
                type=event_type,
                x=point.x,
                y=point.y,
                index=index,
                factors=factors,
                salience=salience,
                level=get_salience_level(salience),
                reason=reason,
            )
        )

    return candidates


def _interval_candidates(series: Series, analysis: SeriesAnalysis) -> List[CandidateEvent]:
    """
    Builds one CandidateEvent for every point-to-point change that
    Stage 4 (sudden changes, a GLOBAL comparison) or Stage 6 (trend
    breaks, a LOCAL comparison) already flagged. Exactly like
    _point_candidates() above, a change that appears in neither list was
    already judged unremarkable, so it's not scored at all.
    """
    n_points = len(series.points)
    value_range = analysis.range.maximum.y - analysis.range.minimum.y

    x_to_index: Dict[str, int] = {p.x: idx for idx, p in enumerate(series.points)}

    sudden_by_to_index = {x_to_index[sc.to_x]: sc for sc in analysis.sudden_changes}
    trend_break_by_to_index = {x_to_index[tb.to_x]: tb for tb in analysis.trend_breaks}

    candidate_to_indices = set(sudden_by_to_index) | set(trend_break_by_to_index)

    candidates: List[CandidateEvent] = []
    for to_index in sorted(candidate_to_indices):
        from_index = to_index - 1
        from_point = series.points[from_index]
        to_point = series.points[to_index]

        sudden = sudden_by_to_index.get(to_index)
        trend_break = trend_break_by_to_index.get(to_index)

        absolute_change = sudden.absolute_change if sudden else trend_break.absolute_change
        percentage_change = sudden.percentage_change if sudden else None
        event_type = "increase" if absolute_change > 0 else "decrease"

        # Magnitude uses the same "relative to this chart's own y-range"
        # idea as _point_candidates(), so point and interval magnitudes
        # stay on a comparable scale instead of mixing units.
        magnitude = _clip01(abs(absolute_change) / value_range) if value_range > 0 else 0.0
        unusualness = _clip01(abs(sudden.robust_z_score) / Z_SCORE_SATURATION_REFERENCE) if sudden else 0.0
        trend_disruption = (
            _clip01(abs(trend_break.local_robust_z_score) / Z_SCORE_SATURATION_REFERENCE) if trend_break else 0.0
        )
        position_importance = round(to_index / (n_points - 1), 4) if n_points > 1 else 0.0
        confidence = min(from_point.confidence, to_point.confidence)

        factors = SalienceFactors(
            magnitude=round(magnitude, 4),
            unusualness=round(unusualness, 4),
            trend_disruption=round(trend_disruption, 4),
            position_importance=position_importance,
            confidence=confidence,
        )
        salience = _combine(factors)

        reasons = []
        if sudden:
            reasons.append(f"unusual compared to every other change in the chart (robust z-score {sudden.robust_z_score})")
        if trend_break:
            reasons.append(f"a break from the recently established local pattern (local z-score {trend_break.local_robust_z_score})")
        reason = (
            f"{from_point.x} to {to_point.x} ({absolute_change:+g}) is " + ", and ".join(reasons) + "."
        )

        candidates.append(
            CandidateEvent(
                type=event_type,
                from_x=from_point.x,
                to_x=to_point.x,
                absolute_change=absolute_change,
                percentage_change=percentage_change,
                index=to_index,
                factors=factors,
                salience=salience,
                level=get_salience_level(salience),
                reason=reason,
            )
        )

    return candidates


def score_series_salience(series: Series, analysis: SeriesAnalysis) -> List[CandidateEvent]:
    """
    STAGE 8 entry point for ONE series.

    Gathers every candidate the earlier stages already flagged (points
    from local extrema / turning points / outliers, intervals from
    sudden changes / trend breaks), scores each one, and returns ALL of
    them sorted by salience, highest first.

    Deliberately returns EVERY scored candidate rather than only the
    "important" ones -- deciding the cutoff (the "3-7 things that
    actually get narrated") is Stage 9's job. Keeping scoring and
    filtering as two separate steps means we can retune the filter later
    without recomputing a single score.
    """
    candidates = _point_candidates(series, analysis) + _interval_candidates(series, analysis)
    candidates.sort(key=lambda c: c.salience, reverse=True)
    return candidates


def filter_meaningful_events(
    ranked_events: List[CandidateEvent],
    max_count: int = EXTENDED_MAX_EVENTS,
) -> List[CandidateEvent]:
    """
    STAGE 9: drops "negligible" events entirely, then keeps at most
    max_count of what's left.

    This assumes ranked_events is ALREADY sorted highest-salience-first
    (exactly what score_series_salience() returns), so "keep the first
    max_count" is the same as "keep the highest-scoring max_count."

    We do NOT pad the result back up to max_count with negligible events
    if there aren't enough meaningful ones -- a chart with only one real
    story to tell should report one event, not manufacture two more just
    to hit a target count. Honesty about "there isn't much else going on
    here" is itself useful information.
    """
    meaningful = [event for event in ranked_events if event.level != NEGLIGIBLE_LEVEL]
    return meaningful[:max_count]


def filter_headline_events(
    ranked_events: List[CandidateEvent],
    max_count: int = HEADLINE_MAX_EVENTS,
) -> List[CandidateEvent]:
    """
    STAGE 9: the short "default narration" list -- at most max_count
    events, always a subset of filter_meaningful_events() (so the
    headline list can never include something the extended list excluded).
    """
    return filter_meaningful_events(ranked_events, max_count=EXTENDED_MAX_EVENTS)[:max_count]


def _counts_by_level(ranked_events: List[CandidateEvent]) -> Dict[str, int]:
    """
    Tally how many events fell into each salience band. Starts every
    level at 0 (rather than only including levels that actually
    occurred), so a caller can always safely read
    counts_by_level["critical"] without a KeyError, even when a chart
    has zero critical events.
    """
    counts = {level: 0 for level in ALL_SALIENCE_LEVELS}
    for event in ranked_events:
        counts[event.level] += 1
    return counts


def build_salience_summary(chart_id: str, series: Series, analysis: SeriesAnalysis) -> SalienceSummary:
    """
    STAGE 9 entry point: runs Stage 8's scoring, then applies both
    filtering tiers, and packages everything (headline / extended / full
    / counts) into one SalienceSummary for this series.

    This is what Person 3's narration layer is actually meant to call --
    score_series_salience() alone (Stage 8) is still available for
    anyone who wants the raw ranked list without any filtering opinion
    applied on top.
    """
    ranked_events = score_series_salience(series, analysis)

    return SalienceSummary(
        chart_id=chart_id,
        series_id=series.id,
        headline_events=filter_headline_events(ranked_events),
        extended_events=filter_meaningful_events(ranked_events),
        all_events=ranked_events,
        counts_by_level=_counts_by_level(ranked_events),
    )