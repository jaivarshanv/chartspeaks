"""
narrator.py
-----------
STAGE 10: turning verified facts into natural language -- "Brain B".

EVERYTHING BEFORE THIS FILE (analyzer.py, salience.py) is "Brain A":
deterministic math, no LLM involved anywhere, the single source of truth
for every number ChartSpeak ever states. This file adds "Brain B": an
optional LLM that phrases Brain A's facts more naturally for speech.

THE CORE RULE (project spec, stated from day one): Brain B may explain
verified facts. It may NEVER invent a number, a trend, or an event that
Brain A didn't already compute and verify. This file enforces that rule
in TWO independent ways, so a failure of one doesn't mean the whole
system is undefended:

1. WHAT THE LLM IS SHOWN. render_llm_narration() never hands the LLM the
   raw CSR, the full AnalysisResult, or the full SalienceSummary. It only
   ever sees a NarrationFacts object (models.py) -- a small, fixed,
   whitelisted set of fields. If a number isn't on that model, there is
   no way for it to end up in the prompt at all.

2. WHAT THE LLM PRODUCES IS CHECKED AFTERWARDS. is_grounded() scans the
   LLM's own output text for every number it contains, and rejects the
   whole narration unless every single one of those numbers can be
   traced back to NarrationFacts. If that check fails for any reason --
   the LLM hallucinated a number, miscopied one, or did its own
   arithmetic -- we do NOT show the user a "maybe correct" narration. We
   fall back to render_template_narration(), which is pure string
   formatting over the same NarrationFacts and therefore CANNOT contain
   an invented number, by construction.

3. IF THE LLM IS UNAVAILABLE AT ALL -- no ANTHROPIC_API_KEY set, the
   `anthropic` package isn't installed, no network, or the API call
   errors -- render_llm_narration() returns None and narrate() silently
   falls back to the template. This means the whole pipeline (Stages
   1-10) works right now, offline, with no API key, which matters for a
   hackathon demo where wifi is not guaranteed. Add an API key later and
   the exact same code automatically starts using the LLM -- nothing
   else has to change.

*** HONESTY NOTE *** The grounding check in this file is a heuristic --
it compares strings of digits, not meanings. It cannot catch an LLM that
reuses a real number but attaches it to the wrong event, or that phrases
something misleadingly without using any new digits at all. It is a real
safety net, not a proof of correctness. Treat it as one layer of defense,
not the only one.
"""

import os
import re
from typing import Optional, Set

from models import ChartCSR, Series, SeriesAnalysis, SalienceSummary, CandidateEvent
from models import NarrationFacts, NarrationResult


def build_narration_facts(
    csr: ChartCSR,
    series: Series,
    series_analysis: SeriesAnalysis,
    summary: SalienceSummary,
) -> NarrationFacts:
    """
    Assembles the ONLY data narrator.py is allowed to work with, by
    copying a fixed, small set of fields out of the CSR, the analysis,
    and the salience summary. Nothing is computed in this function --
    every value here already came from analyzer.py or salience.py.
    """
    return NarrationFacts(
        chart_id=csr.chart_id,
        series_id=series.id,
        chart_title=csr.title,
        chart_type=csr.chart_type,
        series_name=series.name,
        x_label=csr.x_axis.label,
        x_unit=csr.x_axis.unit,
        x_first=series.points[0].x,
        x_last=series.points[-1].x,
        y_label=csr.y_axis.label,
        y_unit=csr.y_axis.unit,
        trend_direction=series_analysis.overall_trend.direction,
        trend_slope=series_analysis.overall_trend.slope,
        trend_strength=series_analysis.overall_trend.strength,
        range_min=series_analysis.range.minimum,
        range_max=series_analysis.range.maximum,
        headline_events=summary.headline_events,
    )


# =========================================================
# PART A: THE TEMPLATE NARRATOR (no LLM, cannot hallucinate)
# =========================================================

def _unit_suffix(unit: Optional[str]) -> str:
    return f" {unit}" if unit else ""


def _describe_trend_strength(strength: float) -> str:
    """
    Turns the 0-1 correlation-based strength number into a plain-English
    qualifier. The THRESHOLDS here (0.7, 0.4) are our own reasonable
    first guess, exactly like the salience level bands in salience.py --
    not scientifically tuned, easy to change.
    """
    if strength >= 0.7:
        return "a clear and consistent"
    if strength >= 0.4:
        return "a moderate"
    return "a weak, inconsistent"


def _describe_event(event: CandidateEvent, series_name: str, y_unit: Optional[str]) -> str:
    """
    Speech-friendly phrasing for one headline event. Deliberately NOT the
    same text as CandidateEvent.reason in salience.py -- that field was
    written for debugging (it mentions "robust z-score", which is not
    something you want read aloud to a blind user). This function only
    ever uses fields already present on the event: x, y, from_x, to_x,
    absolute_change, percentage_change. It adds no new numbers.
    """
    unit = _unit_suffix(y_unit)
    if event.type in ("peak", "valley"):
        word = "highest" if event.type == "peak" else "lowest"
        return f"At {event.x}, {series_name} reached its {word} notable point, {event.y:g}{unit}."
    else:
        direction_word = "rose" if event.type == "increase" else "fell"
        pct = f" ({abs(event.percentage_change):g}%)" if event.percentage_change is not None else ""
        return (
            f"Between {event.from_x} and {event.to_x}, {series_name} {direction_word} "
            f"by {abs(event.absolute_change):g}{unit}{pct}."
        )


def render_template_narration(facts: NarrationFacts) -> str:
    """
    Pure string formatting over NarrationFacts -- zero LLM calls. Because
    every sentence here is built directly from a field on `facts`, this
    function is INCAPABLE of stating a number that Brain A didn't
    compute. This is what ChartSpeak says today, offline, with no API
    key -- and what it always falls back to if the LLM path fails or
    fails its grounding check.
    """
    x_unit = _unit_suffix(facts.x_unit)
    sentences = [
        f"This is '{facts.chart_title}', a {facts.chart_type} chart showing "
        f"{facts.series_name} against {facts.x_label}{x_unit} from {facts.x_first} to {facts.x_last}."
    ]

    strength_phrase = _describe_trend_strength(facts.trend_strength)
    sentences.append(
        f"Overall, {facts.series_name} is {facts.trend_direction}, following {strength_phrase} pattern."
    )

    if not facts.headline_events:
        sentences.append(
            "There is no single standout event in this chart -- it stays close to its overall pattern throughout."
        )
    else:
        sentences.append("The most notable points are:")
        for event in facts.headline_events:
            sentences.append(_describe_event(event, facts.series_name, facts.y_unit))

    return " ".join(sentences)


# =========================================================
# PART B: THE LLM NARRATOR (optional, must pass grounding)
# =========================================================

NARRATION_SYSTEM_PROMPT = """You write short spoken narration for a chart-accessibility tool used by blind and deafblind users.

You will be given a fixed list of verified facts about one chart. Rewrite them as 2-4 natural, clear sentences suitable for text-to-speech.

STRICT RULES:
1. Use ONLY the numbers, labels, and words given below. Do not calculate, estimate, round differently, convert units, or invent any number, date, or fact that is not explicitly listed.
2. Do not add opinions, guesses, or commentary about causes.
3. Mention every event listed, in the order given -- do not skip any, do not add extra ones.
4. Keep it concise. No headers, no bullet points, no markdown -- plain spoken sentences only."""


def _facts_to_prompt(facts: NarrationFacts) -> str:
    lines = [
        f"Chart title: {facts.chart_title}",
        f"Chart type: {facts.chart_type}",
        f"Series name: {facts.series_name}",
        f"X axis: {facts.x_label}" + (f" ({facts.x_unit})" if facts.x_unit else ""),
        f"X range: {facts.x_first} to {facts.x_last}",
        f"Y axis: {facts.y_label}" + (f" ({facts.y_unit})" if facts.y_unit else ""),
        f"Overall trend: {facts.trend_direction}, slope {facts.trend_slope:g}, strength {facts.trend_strength:g} (0-1 scale)",
        f"Minimum: {facts.range_min.y:g} at {facts.range_min.x}",
        f"Maximum: {facts.range_max.y:g} at {facts.range_max.x}",
    ]
    if not facts.headline_events:
        lines.append("Notable events: none -- nothing rose above the 'negligible' salience level.")
    else:
        lines.append(f"Notable events, in order of importance ({len(facts.headline_events)} total):")
        for i, event in enumerate(facts.headline_events, start=1):
            if event.type in ("peak", "valley"):
                lines.append(f"  {i}. [{event.level}] {event.type} at {event.x}, value {event.y:g}")
            else:
                lines.append(
                    f"  {i}. [{event.level}] {event.type} from {event.from_x} to {event.to_x}, "
                    f"change {event.absolute_change:+g}"
                    + (f" ({event.percentage_change:+g}%)" if event.percentage_change is not None else "")
                )
    return "\n".join(lines)


def render_llm_narration(facts: NarrationFacts, model: str = "claude-sonnet-4-5") -> Optional[str]:
    """
    Calls an LLM to phrase `facts` naturally. Returns None (never raises)
    if the LLM path isn't available for ANY reason -- missing package,
    missing API key, network error, API error -- so callers can always
    safely fall back to the template. This function does not decide
    whether the result is trustworthy; narrate() below does that with
    is_grounded().
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=NARRATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _facts_to_prompt(facts)}],
        )
        return response.content[0].text.strip()
    except Exception as error:
        print(f"[narrator] LLM call failed ({error}); falling back to template narration.")
        return None


# =========================================================
# PART C: THE GROUNDING CHECK (catches LLM hallucination)
# =========================================================

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def _extract_numbers(text: str) -> Set[str]:
    """Every digit-sequence the LLM's text contains, as plain strings."""
    return set(_NUMBER_PATTERN.findall(text))


def _allowed_number_strings(value) -> Set[str]:
    """
    All the ways a single numeric fact might reasonably be printed by an
    LLM (as an integer, with one decimal, without a trailing .0, without
    its sign). This exists so a true fact isn't rejected just because the
    LLM wrote "25" instead of "25.0" -- but it never adds a NEW value,
    only alternate spellings of the SAME value.
    """
    if value is None:
        return set()
    try:
        f = float(value)
    except (TypeError, ValueError):
        return set()
    out = {str(f), str(abs(f))}
    if f == int(f):
        out.add(str(int(f)))
        out.add(str(abs(int(f))))
    out.add(f"{f:.1f}")
    out.add(f"{abs(f):.1f}")
    return out


def _facts_allowed_numbers(facts: NarrationFacts) -> Set[str]:
    """Every number the LLM is allowed to have used, gathered from NarrationFacts only."""
    allowed: Set[str] = set()
    for value in (facts.trend_slope, facts.trend_strength, facts.range_min.y, facts.range_max.y):
        allowed |= _allowed_number_strings(value)
    for event in facts.headline_events:
        for value in (event.y, event.absolute_change, event.percentage_change):
            allowed |= _allowed_number_strings(value)
    return allowed


def is_grounded(text: str, facts: NarrationFacts):
    """
    Returns (True, []) if every number in `text` traces back to `facts`.
    Otherwise returns (False, [numbers found in text that don't trace
    back]) so the caller (and a developer debugging this) can see exactly
    what triggered the rejection.
    """
    found = _extract_numbers(text)
    allowed = _facts_allowed_numbers(facts)
    ungrounded = sorted(found - allowed)
    return (len(ungrounded) == 0, ungrounded)


# =========================================================
# PART D: THE ORCHESTRATOR -- what everything else calls
# =========================================================

def narrate(facts: NarrationFacts, prefer_llm: bool = True) -> NarrationResult:
    """
    STAGE 10 entry point. Tries the LLM (only if prefer_llm=True AND an
    API key is configured); if that produces text AND it passes
    is_grounded(), returns it with source="llm". In every other case --
    no key, no package, API error, or a grounding failure -- returns the
    template narration instead, which is always available and always
    grounded by construction.
    """
    if prefer_llm:
        llm_text = render_llm_narration(facts)
        if llm_text is not None:
            grounded, ungrounded_numbers = is_grounded(llm_text, facts)
            if grounded:
                return NarrationResult(
                    chart_id=facts.chart_id,
                    series_id=facts.series_id,
                    text=llm_text,
                    source="llm",
                    grounded=True,
                )
            print(f"[narrator] Rejected LLM narration -- ungrounded numbers found: {ungrounded_numbers}")

    return NarrationResult(
        chart_id=facts.chart_id,
        series_id=facts.series_id,
        text=render_template_narration(facts),
        source="template",
        grounded=True,
    )