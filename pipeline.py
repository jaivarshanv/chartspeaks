"""
pipeline.py
-----------
STAGE 11: the one shared "run everything" function.

Through Stage 10, main.py itself contained the glue code that calls
analyze_chart() -> build_salience_summary() -> build_narration_facts()
-> narrate() for every series in a chart. Now that api.py ALSO needs to
run that exact same sequence (for a chart that arrived over HTTP instead
of from test_data.py), that glue code needs to live in exactly one
place -- otherwise main.py and api.py could quietly drift apart over
time (e.g. someone fixes a bug in one copy and forgets the other).

run_pipeline() is that one place. Both main.py (the command-line demo)
and api.py (the HTTP endpoint) call it now. Neither of them re-implements
any analysis, salience, or narration logic themselves.
"""

from models import ChartCSR, ChartSpeakResponse
from analyzer import analyze_chart
from salience import build_salience_summary
from narrator import build_narration_facts, narrate


def run_pipeline(csr: ChartCSR, prefer_llm: bool = True) -> ChartSpeakResponse:
    """
    Runs the full ChartSpeak pipeline for one chart, end to end:
        CSR -> AnalysisResult (Stages 1-7)
            -> one SalienceSummary per series (Stages 8-9)
            -> one NarrationResult per series (Stage 10)
    and returns everything bundled into a single ChartSpeakResponse.

    prefer_llm is passed straight through to narrate() -- see narrator.py
    for why this defaults to True but safely falls back to the template
    narrator whenever no API key is configured.
    """
    analysis = analyze_chart(csr)

    salience_summaries = []
    narrations = []

    for series, series_analysis in zip(csr.series, analysis.series):
        summary = build_salience_summary(csr.chart_id, series, series_analysis)
        salience_summaries.append(summary)

        facts = build_narration_facts(csr, series, series_analysis, summary)
        narrations.append(narrate(facts, prefer_llm=prefer_llm))

    return ChartSpeakResponse(
        chart_id=csr.chart_id,
        analysis=analysis,
        salience_summaries=salience_summaries,
        narrations=narrations,
    )