"""
main.py
-------
The entry point. This is the file you actually run.

It does three things, in order:
1. Load the fake CSR test data (test_data.py)
2. Run it through the analysis engine (analyzer.py)
3. Print the result as nicely formatted JSON

WHAT IS JSON?
JSON (JavaScript Object Notation) is just a plain-text way of writing
structured data (numbers, text, lists, nested groups) that almost every
programming language and web API can read and write. It's the format
our AnalysisResult will eventually travel over the network in, when
Person 3's app calls our API (Stage 11).
"""

import json

from test_data import get_fake_csr, get_fake_multi_series_csr
from analyzer import analyze_chart
from salience import build_salience_summary


def run_and_print(csr, label: str):
    """Small shared helper so we don't repeat the same lines for each test chart."""
    print(f"=== {label} ===")
    print(f"Loaded chart: '{csr.title}' (chart_id={csr.chart_id})")
    print(f"Series: {', '.join(s.name for s in csr.series)}")
    print("-" * 50)

    result = analyze_chart(csr)

    # by_alias=True makes PointChange (and the other "from"/"to" models)
    # print as "from"/"to" instead of "from_x"/"to_x" -- see models.py.
    result_dict = result.model_dump(by_alias=True)
    print(json.dumps(result_dict, indent=2))

    # STAGE 8 + 9: score every candidate event, then filter it down to
    # what a narration layer would actually use. build_salience_summary()
    # is the function Person 3's side is meant to call -- it wraps
    # score_series_salience() (still available on its own if anyone
    # wants the unfiltered ranked list).
    for series, series_analysis in zip(csr.series, result.series):
        print(f"\n--- Salience summary for '{series.name}' ---")
        summary = build_salience_summary(csr.chart_id, series, series_analysis)

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