"""
api.py
------
STAGE 11: the HTTP API.

This is what turns ChartSpeak from "a script I run in a terminal" into
"a service Person 3's mobile app can actually call." Nothing about the
analysis, salience, or narration logic changes here -- this file's only
job is to accept a CSR over HTTP and hand back a ChartSpeakResponse over
HTTP, using run_pipeline() (pipeline.py) to do the actual work.

WHAT IS FASTAPI?
FastAPI is a Python web framework built directly on top of Pydantic --
the same library our models.py has used since Stage 1. That's exactly
why we chose it: a FastAPI endpoint can take a Pydantic model (like our
ChartCSR) as its input type, and FastAPI will automatically:
  1. read the incoming JSON request body,
  2. validate it against that model (reusing the exact same validation
     rules ChartCSR has always had -- no new code),
  3. reject it with a clear 422 error, automatically, if it doesn't
     match the schema, before our function body even runs,
  4. and, on the way out, serialize whatever Pydantic model we return
     back into JSON.
So the "contract" between Person 1's real pipeline (or Person 3's app)
and this service is still just the CSR / ChartSpeakResponse schemas in
models.py -- nothing new to maintain.

HOW TO RUN THIS FILE:
Unlike main.py, you don't run this with `python api.py`. You run it with
a separate program called an ASGI server, which is what actually listens
for network requests and hands them to FastAPI:

    uvicorn api:app --reload

    "api"      = this file, api.py
    "app"      = the FastAPI() object defined below
    "--reload" = automatically restarts the server when you save a code
                 change, which is convenient while developing

Once it's running, FastAPI automatically builds an interactive docs page
at http://127.0.0.1:8000/docs -- you can open that in a browser and try
the /analyze endpoint by hand, no curl or Postman required.
"""

from fastapi import FastAPI, HTTPException

from models import ChartCSR, ChartSpeakResponse
from pipeline import run_pipeline

app = FastAPI(
    title="ChartSpeak Analysis API",
    description=(
        "Accepts a Chart Semantic Representation (CSR) and returns verified "
        "mathematical analysis, ranked salient events, and narration."
    ),
    version="0.1.0",
)


@app.get("/health")
def health():
    """
    A trivial endpoint with no real logic -- just something Person 3's
    app (or a deploy script) can call to check "is the service up at
    all?" before trying to send it real chart data.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=ChartSpeakResponse)
def analyze(csr: ChartCSR) -> ChartSpeakResponse:
    """
    The one real endpoint.

    FastAPI has already validated the incoming JSON against ChartCSR by
    the time this function body runs -- if you send it something that
    doesn't match the schema (a missing field, a string where a number
    was expected), the caller gets a 422 error back automatically and
    this function is never even called.

    Everything this function does is call the SAME run_pipeline()
    function main.py calls. If you've verified main.py's output for a
    given chart, you already know what this endpoint will return for
    that same chart.
    """
    try:
        return run_pipeline(csr)
    except Exception as error:
        # A defensive fallback for anything unexpected in analysis itself
        # (which we don't expect, since analyze_chart etc. are already
        # verified) -- so a caller gets a clear 500 error instead of the
        # server just crashing.
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}")