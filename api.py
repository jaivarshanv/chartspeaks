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

import json
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from models import ChartCSR, ChartSpeakResponse
from pipeline import run_pipeline
from chart4blind_adapter import parse_chart4blind_csv
from image_extractor import extract_csr, ExtractionError

# STAGE 14 (frontend integration): temporary debugging logs so a real
# upload's progress through the pipeline is visible on the server
# console -- "did we even receive the file, what did OCR/extraction
# actually do." Deliberately logs metadata only (filename, size, byte
# count, stage reached) -- never the image's own pixel/byte contents.
logging.basicConfig(level=logging.INFO, format="[chartspeak] %(message)s")
logger = logging.getLogger("chartspeak")

app = FastAPI(
    title="ChartSpeak Analysis API",
    description=(
        "Accepts a Chart Semantic Representation (CSR) and returns verified "
        "mathematical analysis, ranked salient events, and narration."
    ),
    version="0.1.0",
)

# STAGE 13 (frontend integration): the React app runs on a different origin
# (the Vite dev server, e.g. http://localhost:5173) than this API
# (http://localhost:8000), so the browser will block the request unless we
# explicitly allow it via CORS. CHARTSPEAK_CORS_ORIGINS lets you restrict
# this to a real domain in production; it defaults to "*" (any origin) so
# the hackathon demo works with zero configuration out of the box.
_cors_origins = os.environ.get("CHARTSPEAK_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else [o.strip() for o in _cors_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


# ---------------------------------------------------------------------
# STAGE 13/14 (frontend integration): /analyze-file
#
# The frontend's upload area naturally wants to send a FILE (what the
# user actually picked), not a hand-built ChartCSR JSON body. This
# endpoint is an honest bridge for that:
#
#   .json          -> parsed directly as a ChartCSR (Person 1's
#                      pipeline, or anything else, can export this).
#   .csv           -> parsed with chart4blind_adapter.parse_chart4blind_csv()
#                      (Stage 12) -- Chart4Blind's existing "Export as
#                      CSV" output can be fed straight in.
#   .png/.jpg/.jpeg -> run through image_extractor.extract_csr() (Stage 14):
#                      a real, deterministic (OCR + computer vision,
#                      NOT an LLM) chart digitizer -- see that file's
#                      module docstring for exactly how it works and
#                      its honest scope/limitations. If it can't
#                      confidently read the chart, it raises
#                      ExtractionError with a specific reason, which
#                      becomes this endpoint's 422 response -- never a
#                      made-up result.
# ---------------------------------------------------------------------
SUPPORTED_DATA_EXTENSIONS = (".json", ".csv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
UNSUPPORTED_IMAGE_EXTENSIONS = (".pdf", ".webp", ".svg", ".gif", ".bmp")


@app.post("/analyze-file", response_model=ChartSpeakResponse)
async def analyze_file(file: UploadFile = File(...)) -> ChartSpeakResponse:
    filename = (file.filename or "").lower()
    raw_bytes = await file.read()
    logger.info(
        "received upload: filename=%r content_type=%r bytes=%d",
        file.filename, file.content_type, len(raw_bytes),
    )

    if not raw_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    if filename.endswith(IMAGE_EXTENSIONS):
        logger.info("dispatching to image_extractor (OCR + CV extraction) for %r", file.filename)
        try:
            csr = extract_csr(raw_bytes, chart_id=filename.rsplit(".", 1)[0] or "uploaded_chart")
        except ExtractionError as error:
            logger.info("extraction failed for %r: %s", file.filename, error)
            raise HTTPException(status_code=422, detail=str(error))
        logger.info(
            "ChartCSR generated for %r: %d series, %d total points",
            file.filename, len(csr.series), sum(len(s.points) for s in csr.series),
        )
        try:
            result = run_pipeline(csr)
            logger.info("analysis pipeline complete for %r", file.filename)
            return result
        except Exception as error:
            logger.exception("analysis pipeline raised for %r", file.filename)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error}")

    if filename.endswith(UNSUPPORTED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=(
                "ChartSpeak can read PNG and JPG chart images directly. "
                f"{filename.rsplit('.', 1)[-1].upper()} isn't supported yet -- "
                "try exporting as PNG/JPG, or upload a .json (ChartCSR) or "
                ".csv (Chart4Blind export) file instead."
            ),
        )

    if not filename.endswith(SUPPORTED_DATA_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a PNG/JPG chart image, a .json (ChartCSR), or a .csv (Chart4Blind export) file.",
        )

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="The file isn't readable text -- it may be corrupted.")

    try:
        if filename.endswith(".json"):
            csr = ChartCSR.model_validate(json.loads(text))
        else:
            csr = parse_chart4blind_csv(text, chart_id=filename.rsplit(".", 1)[0] or "uploaded_chart")
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="That .json file isn't valid JSON.")
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=f"That file doesn't match the expected chart format: {error}")

    if not csr.series or all(len(s.points) == 0 for s in csr.series):
        raise HTTPException(status_code=422, detail="ChartSpeak couldn't find any data points in this chart.")

    try:
        return run_pipeline(csr)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}")
