# test_csr_sample.py
# Quick manual check: build one hard-coded CSR object by hand,
# just to confirm the schema shape works end-to-end.
# This is NOT a real extraction yet — it's fake/sample data,
# same idea as the old "testCSR" object in the TypeScript version.
from dataclasses import asdict
import json
from src.schema.csr import (
    ChartSemanticRepresentation,
    ChartAxis,
    ChartSeries,
    ChartPoint,
)

test_csr = ChartSemanticRepresentation(
    chart_type="line",
    title="Test Sales Chart",
    x_axis=ChartAxis(
        label="Month",
        scale_type="linear",
        unit="",
        values=["1", "2", "3", "4", "5"],
    ),
    y_axis=ChartAxis(
        label="Sales",
        scale_type="linear",
        unit="units",
        values=["10", "20", "30", "40", "50"],
    ),
    series=[
        ChartSeries(
            name="Sales",
            points=[
                ChartPoint(x=1, y=10),
                ChartPoint(x=2, y=20),
                ChartPoint(x=3, y=15),
                ChartPoint(x=4, y=40),
                ChartPoint(x=5, y=35),
            ],
        )
    ],
)

# For now, just print it so we can visually confirm it built correctly.
# Convert the dataclass object into a plain dictionary...
csr_dict = asdict(test_csr)

# ...then convert that dictionary into a JSON string.
# indent=2 just makes it pretty-printed / readable, not a functional requirement.
csr_json = json.dumps(csr_dict, indent=2)

# Write the JSON to a file inside data/outputs, instead of just printing it.
output_path = "data/outputs/sample_csr.json"

with open(output_path, "w") as f:
    f.write(csr_json)

print(f"CSR JSON written to: {output_path}")