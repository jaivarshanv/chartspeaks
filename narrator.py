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


# Lead-ins that open each event sentence, so the fallback narration doesn't
# start every clause the same way. Cycled BY INDEX, never randomly, so the
# same facts always produce the same words -- a random narrator would be
# untestable and would read differently on every replay of the same chart.
_EVENT_LEAD_INS = (
    "The sharpest change comes here.",
    "From there,",
    "Also worth noting,",
    "And",
)


def _describe_event(
    event: CandidateEvent,
    series_name: str,
    y_unit: Optional[str],
    position: int = 0,
) -> str:
    """
    Speech-friendly phrasing for one headline event. Deliberately NOT the
    same text as CandidateEvent.reason in salience.py -- that field was
    written for debugging (it mentions "robust z-score", which is not
    something you want read aloud to a blind user). This function only
    ever uses fields already present on the event: x, y, from_x, to_x,
    absolute_change, percentage_change. It adds no new numbers.

    `position` selects a lead-in phrase so consecutive event sentences
    don't all open identically; it never changes any stated fact.
    """
    unit = _unit_suffix(y_unit)
    lead = _EVENT_LEAD_INS[position % len(_EVENT_LEAD_INS)]

    if event.type in ("peak", "valley"):
        word = "highest" if event.type == "peak" else "lowest"
        body = f"at {event.x}, {series_name} reached its {word} notable point, {event.y:g}{unit}"
    else:
        direction_word = "rose" if event.type == "increase" else "fell"
        pct = f", or {abs(event.percentage_change):g}%" if event.percentage_change is not None else ""
        body = (
            f"between {event.from_x} and {event.to_x}, {series_name} {direction_word} "
            f"by {abs(event.absolute_change):g}{unit}{pct}"
        )

    if lead.endswith("."):
        return f"{lead[:-1]}: {body}."
    return f"{lead} {body}."


def render_template_narration(facts: NarrationFacts) -> str:
    """
    Pure string formatting over NarrationFacts -- zero LLM calls. Because
    every sentence here is built directly from a field on `facts`, this
    function is INCAPABLE of stating a number that Brain A didn't
    compute. This is what ChartSpeak says today, offline, with no API
    key -- and what it always falls back to if the LLM path fails or
    fails its grounding check. It follows the same spoken contract as
    NARRATION_SYSTEM_PROMPT: identify, trend, range, then every headline
    event in the order salience.py ranked them.
    """
    x_unit = _unit_suffix(facts.x_unit)
    y_unit = _unit_suffix(facts.y_unit)

    sentences = [
        f"'{facts.chart_title}' is a {facts.chart_type} chart tracking {facts.series_name} "
        f"across {facts.x_label}{x_unit}, from {facts.x_first} to {facts.x_last}."
    ]

    strength_phrase = _describe_trend_strength(facts.trend_strength)
    sentences.append(
        f"Across that span the {facts.y_label} is {facts.trend_direction}, in {strength_phrase} pattern, "
        f"moving between a low of {facts.range_min.y:g}{y_unit} at {facts.range_min.x} "
        f"and a high of {facts.range_max.y:g}{y_unit} at {facts.range_max.x}."
    )

    if not facts.headline_events:
        sentences.append(
            "Nothing here rose above a negligible salience level, so the series stays close "
            "to that overall pattern throughout."
        )
    else:
        for position, event in enumerate(facts.headline_events):
            sentences.append(_describe_event(event, facts.series_name, facts.y_unit, position))

    return " ".join(sentences)


# =========================================================
# PART B: THE LLM NARRATOR (optional, must pass grounding)
# =========================================================

NARRATION_SYSTEM_PROMPT = """You are the spoken narrator for a chart-accessibility tool. You turn a verified graph study into natural language that a text-to-speech engine, a screen reader, or a refreshable Braille display can deliver without visual context. The listener cannot see the chart. They hear or feel only what you write. You are not an analyst inventing insights. You are a careful reader of a finished study.

AUDIENCE
Write for blind, low-vision, and deafblind listeners. Assume no sight of color, markers, legends, or layout. Never say "as you can see", "shown on the right", "the blue line", "the highlighted point", or any other phrase that requires vision. Use the names, axis labels, and x-values given in the facts. Keep sentences short enough that a screen reader can pause at the periods. Avoid stacked clauses, parenthetical asides, slashes, tildes, and abbreviations a speech engine will mangle. Never use table language such as "row" or "column".

WHAT YOU MUST COVER
Cover all of the following. This is a coverage checklist, not a script, and not a required sentence order.
1. Identify the chart: the title, the chart type, the series name, what the x-axis measures, the x range from the first value to the last value, and the y-axis name with its unit if a unit is given.
2. The overall trend, in plain language, using the direction given. You may use a qualitative cue the facts already imply, such as clear, moderate, or weak. Never invent a new numeric strength and never recast the slope as a different number. If you mention slope or strength at all, copy the digits exactly.
3. The range: the minimum value and where it occurs, and the maximum value and where it occurs, using the exact figures supplied.
4. Every notable event in the list, in the exact order given, without skipping, merging, splitting, or adding events. For a peak or valley, give the type, the x location, and the value. For an increase or decrease, give the span from the given start x to the given end x, and the change as printed in the facts. Include a percent change only if the facts include one.
5. If the facts say there are no notable events, say clearly that nothing rose above a negligible salience level and that the series stays close to its overall pattern. Do not invent a dip, spike, or season.

VOICE AND FLOW
The listener should hear a person describing the chart, not a form being read out. Cover every required item, but choose the sentence order and the joins that flow best. Vary how your sentences open instead of starting each one with the series name. Combine naturally related facts into a single sentence, for example the trend together with the span it covers, or the maximum together with the peak event that sits at it, rather than one fact per sentence. Use ordinary connective phrasing such as "from there", "the sharpest move comes", or "by the end" to carry the listener between facts. Let numbers arrive inside sentences rather than as a list of readings. Prefer everyday words: "rose", "fell", "reached its highest notable point", "from ... to ...". Spell out the chart type as given, for example line or bar. Repeat the series name when it keeps a sentence clear rather than leaning on "it" across a long stretch. Put the unit next to the number the way the facts do.

LENGTH AND SHAPE
Two to four complete spoken sentences, or at most six if several headline events must each be named without crowding. One flowing paragraph. No title line, no "Summary:" prefix, no bullet points, no headings, no markdown, no emoji, no JSON, no line breaks for structure.

GROUNDING RULES FOR NUMBERS
This is the non-negotiable contract. Every digit sequence you output must appear in the facts, or be an exact alternate spelling of a fact already given, for example 25 instead of 25.0, or the absolute value of a signed change the facts already stated.
Do not calculate a new total, average, median, ratio, difference, growth rate, or annualized figure. Do not round, truncate, or reformat a number into a new digit sequence: do not turn 12.1 into 12, 12.10, or 12 million unless those exact tokens are in the facts. Do not convert units, and do not restate a unit as thousands, millions, or a different currency. Do not infer a year, month, day, or index that is not written in the facts. Do not count the events yourself and announce a tally unless that count is already in the facts. Do not attach a real number to the wrong event, axis, or location. If a value is missing, for example no percent change, simply omit it. Never fill a gap with an estimate.

GROUNDING RULES FOR LANGUAGE
Do not speculate about why the series moved. Do not mention policy, seasons, markets, weather, product launches, or data-collection errors. Do not hedge with "it seems", "probably", or "the chart suggests a recovery". Do not praise or warn with phrases such as "strong performance" or "alarming drop". Trend direction words that are in the facts, such as increasing, decreasing, or stable, may be used. Salience level tags in the facts, such as high or medium, may be spoken once per event if they help priority, but do not redefine them. Do not compare to other charts, other years, targets, budgets, or normal ranges. Do not mention this tool, APIs, keys, models, prompts, or the analysis pipeline. The listener should hear the chart, not the machinery.

EVENT DISCIPLINE
The notable-events list is already ordered by importance. Speak the events in that given order, not in chronological order, unless the facts are already chronological. Do not promote an unlisted wiggle into an event. Do not drop a listed event because it seems small. Never say "among other changes". If two events share an x value, report both, each with its own verified numbers.

FAILURE MODES TO AVOID
Do not return an empty string. Do not return only the title. Do not paste the facts back as labeled fields. Do not wrap your answer in quotes. Do not apologize, ask follow-up questions, or offer to recalculate. If the facts are sparse, still produce a grammatical identification plus trend plus range, and the required event sentence.

OUTPUT
Return only the narration text: plain sentences, ready to be read aloud. Every number grounded in the facts. Every listed event mentioned in order. Invent nothing."""


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


def _allowed_x_strings(x) -> Set[str]:
    """
    X-axis locations are strings, not floats ("2019", "Q3 2020", "Jan"),
    so _allowed_number_strings() can't read them. The narration is
    REQUIRED to speak them -- "from 2019 to 2024", "at 2022" -- so every
    digit run inside one is allowed. This adds no new value: the digits
    were already handed to the narrator in NarrationFacts.
    """
    if x is None:
        return set()
    return set(_NUMBER_PATTERN.findall(str(x)))


def _facts_allowed_numbers(facts: NarrationFacts) -> Set[str]:
    """Every number the LLM is allowed to have used, gathered from NarrationFacts only."""
    allowed: Set[str] = set()
    for value in (facts.trend_slope, facts.trend_strength, facts.range_min.y, facts.range_max.y):
        allowed |= _allowed_number_strings(value)
    for x in (facts.x_first, facts.x_last, facts.range_min.x, facts.range_max.x):
        allowed |= _allowed_x_strings(x)
    for event in facts.headline_events:
        for value in (event.y, event.absolute_change, event.percentage_change):
            allowed |= _allowed_number_strings(value)
        for x in (event.x, event.from_x, event.to_x):
            allowed |= _allowed_x_strings(x)
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