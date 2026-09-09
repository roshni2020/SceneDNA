"""SceneDNA API + static studio UI.

Single-container deployment target (Cloud Run): FastAPI serves the JSON API under /api and
the compiled Vite frontend from frontend/dist for every other path.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_pipeline import (
    GEMINI_MODEL,
    PRESETS,
    TOTAL_EVENTS_EXPECTED,
    ask_warehouse,
    ensure_mcp,
    extract_scene_dna_from_video,
    gemini_backend,
    gemini_configured,
    extract_scene_dna_mock,
    mcp_clickhouse,
    mcp_status,
    run_pipeline,
)

dotenv.load_dotenv()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024
RESULT_CACHE_TTL = int(os.getenv("RESULT_CACHE_TTL_SEC", "900"))
WARM_PRESETS = os.getenv("WARM_PRESETS", "1") not in ("0", "false", "no")

# Preset results are cached briefly so repeated demo clicks are instant and resilient to Gemini
# overload. Pass ?fresh=1 to force a full agent run.
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def _cached(scene_id: str) -> dict[str, Any] | None:
    with _cache_lock:
        entry = _cache.get(scene_id)
    if not entry or time.time() - entry[0] > RESULT_CACHE_TTL:
        return None
    result = copy.deepcopy(entry[1])
    result["cached"] = True
    result["cached_age_sec"] = round(time.time() - entry[0])
    return result


def _store(scene_id: str, result: dict[str, Any]) -> None:
    if result.get("clickhouse", {}).get("fallback"):
        return
    with _cache_lock:
        _cache[scene_id] = (time.time(), copy.deepcopy(result))


def _warm_presets() -> None:
    for scene_id in PRESETS:
        try:
            _store(scene_id, run_pipeline(extract_scene_dna_mock(scene_id), scene_id, "preset"))
            print(f"[WARM] cached {scene_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARM] {scene_id} failed: {exc}")
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/mpeg"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the MCP server at boot so the first request is fast.
    await run_in_threadpool(ensure_mcp)
    if WARM_PRESETS:
        threading.Thread(target=_warm_presets, name="warm-presets", daemon=True).start()
    yield
    mcp_clickhouse.stop()


app = FastAPI(title="SceneDNA", version="2.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PresetRequest(BaseModel):
    scene_id: str


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(question) > 500:
        raise HTTPException(status_code=413, detail="Question is too long (500 characters max).")
    return await run_in_threadpool(ask_warehouse, question)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    status = mcp_status()
    rows = 0
    if status["connected"]:
        try:
            _, result = await run_in_threadpool(
                mcp_clickhouse.run_query, "SELECT count() FROM viewer_retention_events"
            )
            rows = int(result[0][0]) if result else 0
        except Exception as exc:  # noqa: BLE001
            status["last_error"] = str(exc)[:200]
    return {
        "status": "ok",
        "clickhouse": status["connected"] and rows > 0,
        "rows_scanned": rows,
        "expected_rows": TOTAL_EVENTS_EXPECTED,
        "gemini": gemini_configured(),
        "gemini_backend": gemini_backend(),
        "gemini_model": GEMINI_MODEL,
        "mcp": status,
        "presets": {k: v.scene_title for k, v in PRESETS.items()},
    }


@app.get("/api/mcp/status")
async def mcp_state() -> dict[str, Any]:
    return mcp_status()


@app.post("/api/mcp/reconnect")
async def mcp_reconnect() -> dict[str, Any]:
    await run_in_threadpool(mcp_clickhouse.restart)
    return mcp_status()


@app.post("/api/analyze-preset")
async def analyze_preset(req: PresetRequest, fresh: bool = False) -> dict[str, Any]:
    if req.scene_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset '{req.scene_id}'. Use one of {sorted(PRESETS)}.")
    if not fresh:
        hit = _cached(req.scene_id)
        if hit:
            return hit
    features = extract_scene_dna_mock(req.scene_id)
    result = await run_in_threadpool(run_pipeline, features, req.scene_id, "preset")
    _store(req.scene_id, result)
    return result


@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type '{mime}'. Upload mp4, mov, webm or mkv.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    started = time.perf_counter()
    features = await run_in_threadpool(extract_scene_dna_from_video, data, mime)
    result = await run_in_threadpool(run_pipeline, features, "upload", "video")
    result["agent_trace"].insert(0, {
        "step": 0,
        "actor": "gemini",
        "tool": "gemini.extract_scene_dna",
        "status": "ok",
        "latency_ms": round((time.perf_counter() - started) * 1000 - result["total_latency_ms"], 1),
        "rows": None,
        "detail": f"{file.filename} ({len(data) // 1024} KB, {mime}) -> {features.scene_title}",
    })
    result["upload"] = {"filename": file.filename, "bytes": len(data), "mime": mime}
    return result


# --------------------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------------------
DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")
INDEX_FILE = os.path.join(DIST_DIR, "index.html")

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
    candidate = os.path.join(DIST_DIR, full_path)
    if full_path and os.path.isfile(candidate):
        return FileResponse(candidate)
    if os.path.isfile(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return JSONResponse(status_code=503, content={"detail": "Frontend not built. Run `npm run build` in frontend/."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
