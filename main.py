"""
main.py
-------
The command-line entry point. This is the file you run with
`python main.py` to see ChartSpeak work on the fake test charts.

STAGE 11 CHANGE: this file no longer contains the "call analyze_chart,
then build_salience_summary, then narrate" glue code itself -- that
logic moved into pipeline.py's run_pipeline(), because api.py (the new
HTTP API) needs to run that exact same sequence too. main.py now just
calls run_pipeline() and prints the result nicely, the same way api.py
calls run_pipeline() and returns the result as JSON over HTTP. Every
number printed below is unchanged from Stage 10 -- only where the
orchestration code lives has changed.

WHAT IS JSON?
JSON (JavaScript Object Notation) is just a plain-text way of writing
structured data (numbers, text, lists, nested groups) that almost every
programming language and web API can read and write. It's the format
our AnalysisResult travels over the network in now that Stage 11 has
wrapped it in a real API.
"""

import json

from test_data import get_fake_csr, get_fake_multi_series_csr
from pipeline import run_pipeline


def run_and_print(csr, label: str):
    """Small shared helper so we don't repeat the same lines for each test chart."""
    print(f"=== {label} ===")
    print(f"Loaded chart: '{csr.title}' (chart_id={csr.chart_id})")
    print(f"Series: {', '.join(s.name for s in csr.series)}")
    print("-" * 50)

    # STAGE 11: this one call replaces everything main.py used to do by
    # hand -- analyze_chart(), build_salience_summary() per series, and
    # narrate() per series. api.py calls this exact same function.
    response = run_pipeline(csr)

    # by_alias=True makes PointChange (and the other "from"/"to" models)
    # print as "from"/"to" instead of "from_x"/"to_x" -- see models.py.
    result_dict = response.analysis.model_dump(by_alias=True)
    print(json.dumps(result_dict, indent=2))

    for series, summary, narration in zip(csr.series, response.salience_summaries, response.narrations):
        print(f"\n--- Salience summary for '{series.name}' ---")
        print(f"Counts by level: {summary.counts_by_level}")

        print(f"\nHEADLINE events (default narration, max {len(summary.headline_events)} shown):")
        if not summary.headline_events:
            print("  (nothing rose above 'negligible' -- this chart has no standout story)")
        for event in summary.headline_events:
            print(f"  [{event.level:>9}] {event.salience:.4f}  {event.reason}")

        print(f"\nEXTENDED events (a bit more detail, {len(summary.extended_events)} shown):")
        for event in summary.extended_events:
            print(f"  [{event.level:>9}] {event.salience:.4f}  {event.reason}")

        print(f"\nALL scored candidates ({len(summary.all_events)} total, the 'exact values' layer):")
        for event in summary.all_events:
            print(f"  [{event.level:>9}] {event.salience:.4f}  {event.reason}")

        print(f"\nNARRATION (source={narration.source}, grounded={narration.grounded}):")
        print(f"  {narration.text}")

    print()


def main():
    # STAGE 1-6 example: one series (still works exactly as before --
    # just nested under `series: [...]` since Stage 7 restructured the
    # output. `comparison` will be null, since there's only one series.
    run_and_print(get_fake_csr(), "Single-series chart (Jan-Jun energy)")

    # STAGE 7 example: two series that cross, to exercise the new
    # crossings / growth / convergence comparison logic.
    run_and_print(get_fake_multi_series_csr(), "Two-series chart (A vs B crossing)")


if __name__ == "__main__":
    main()