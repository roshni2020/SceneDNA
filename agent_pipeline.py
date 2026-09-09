"""SceneDNA agent pipeline.

Three stages, all executed at runtime:

1. Feature extraction  - Gemini reads a scene cut (video bytes) and returns structured
                         creative attributes (SceneFeatures). Presets exist for demos.
2. OLAP evidence       - Every analytical query goes through the official ClickHouse MCP
                         server (`mcp-clickhouse`, stdio transport). The pipeline keeps one
                         server process alive and calls its `run_query` tool. Each call is
                         recorded in an agent trace with latency and row counts.
3. Prescription        - Gemini is handed the evidence and a `clickhouse_run_query` tool
                         (routed through the same MCP session) so it can drill down before
                         writing a structured editorial prescription with cut ranges.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from typing import Any, Optional

import dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters, stdio_client

dotenv.load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.6-flash,gemini-flash-latest,gemini-3.1-pro-preview,gemini-2.5-flash",
    ).split(",") if m.strip()
]
TOTAL_EVENTS_EXPECTED = 4_800_000
AGENT_MAX_TOOL_CALLS = int(os.getenv("AGENT_MAX_TOOL_CALLS", "2"))
AGENT_DRILLDOWN = os.getenv("AGENT_DRILLDOWN", "1") not in ("0", "false", "no")


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class SceneFeatures(BaseModel):
    scene_title: str
    duration_sec: int
    dialogue_ratio: float = Field(ge=0.0, le=1.0, description="fraction of runtime with spoken dialogue")
    pacing_wpm: int = Field(description="spoken words per minute")
    motion_intensity: float = Field(ge=0.0, le=1.0, description="camera + subject motion, 0 static to 1 kinetic")
    sentiment: str = Field(description="snake_case framing label, e.g. stagnant_two_shot, kinetic_suspense")
    action_level: float = Field(default=0.0, ge=0.0, le=1.0, description="on-screen physical action, 0 none to 1 constant")
    camera_motion: str = Field(default="low", description="low, medium or high")
    music_intensity: float = Field(default=0.0, ge=0.0, le=1.0, description="score/music prominence, 0 silent to 1 driving")
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0, description="emotional charge of performances, 0 flat to 1 peak")
    avg_shot_length_sec: float = Field(default=0.0, ge=0.0, description="estimated average shot length in seconds")
    visible_characters: int = Field(default=0, ge=0, description="typical number of characters visible on screen")
    summary: str = Field(default="", description="one or two sentence plain-language scene summary")


class CutRecommendation(BaseModel):
    start_sec: int
    end_sec: int
    action: str = Field(description="One of: trim, tighten, insert_cutaway, reframe, reorder")
    rationale: str


class Prescription(BaseModel):
    headline: str
    risk_summary: str
    cuts: list[CutRecommendation]
    camera_directions: list[str]
    predicted_drop_pct_after: float
    confidence: float = Field(ge=0.0, le=1.0)
    report_text: str


def format_timecode(seconds: int | float) -> str:
    seconds = int(max(0, round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------------------
# Stage 1: feature extraction
# --------------------------------------------------------------------------------------
PRESETS: dict[str, SceneFeatures] = {
    "cut_01": SceneFeatures(
        scene_title="Safehouse Dialogue Lock (Cut 3)",
        duration_sec=272,
        dialogue_ratio=0.84,
        pacing_wpm=205,
        motion_intensity=0.14,
        sentiment="stagnant_two_shot",
        action_level=0.08,
        camera_motion="low",
        music_intensity=0.12,
        emotional_intensity=0.55,
        avg_shot_length_sec=6.8,
        visible_characters=2,
        summary="Two operatives argue over a compromised extraction plan in a locked-off two-shot inside a dim safehouse. Little movement, no score, long uninterrupted exchanges.",
    ),
    "cut_02": SceneFeatures(
        scene_title="Warehouse Breach Pre-Assault",
        duration_sec=135,
        dialogue_ratio=0.22,
        pacing_wpm=75,
        motion_intensity=0.76,
        sentiment="kinetic_suspense",
        action_level=0.71,
        camera_motion="high",
        music_intensity=0.68,
        emotional_intensity=0.62,
        avg_shot_length_sec=2.1,
        visible_characters=5,
        summary="A tactical team stacks up outside a warehouse, checks gear and breaches on a countdown. Handheld coverage, driving percussion, short cuts.",
    ),
}


def extract_scene_dna_mock(scene_id: str) -> SceneFeatures:
    return PRESETS.get(scene_id, PRESETS["cut_01"]).model_copy()


def gemini_backend() -> str:
    """'vertex' when Gemini is served through the Google Cloud project (Vertex AI / Gemini
    Enterprise Agent Platform), otherwise 'ai-studio' for a GEMINI_API_KEY."""
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    return "vertex" if use_vertex and os.getenv("GOOGLE_CLOUD_PROJECT") else "ai-studio"


def gemini_configured() -> bool:
    return gemini_backend() == "vertex" or bool(os.getenv("GEMINI_API_KEY"))


def _gemini_client() -> genai.Client:
    if gemini_backend() == "vertex":
        # Uses Application Default Credentials (the Cloud Run service account in production).
        return genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set and Vertex AI is not configured")
    return genai.Client(api_key=api_key)


def _model_chain() -> list[str]:
    chain = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    return chain


def generate_with_fallback(client: genai.Client, contents: Any, config: types.GenerateContentConfig,
                           trace: "AgentTrace | None" = None, label: str = "gemini",
                           chain: list[str] | None = None) -> tuple[Any, str]:
    """Call Gemini, retrying once per model on overload (429/503) before moving down the chain.

    Returns (response, model_id_used)."""
    last_exc: Exception | None = None
    for model_id in (chain or _model_chain()):
        for attempt in range(1):
            try:
                response = client.models.generate_content(model=model_id, contents=contents, config=config)
                return response, model_id
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                message = str(exc)
                transient = any(code in message for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded"))
                if trace is not None:
                    trace.record(f"gemini.{label}", "retry" if transient else "error", 0.0,
                                 f"{model_id}: {message[:140]}", actor="gemini")
                if not transient:
                    break
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"All Gemini models failed: {last_exc}")


def extract_scene_dna_from_video(video_bytes: bytes, mime_type: str) -> SceneFeatures:
    """Multimodal extraction. Falls back to the cut_01 preset if Gemini is unavailable."""
    try:
        client = _gemini_client()
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
        prompt = (
            "You are a post-production analyst. Watch this unreleased scene cut end to end and extract "
            "its creative DNA as structured fields: scene_title (short); duration_sec (integer seconds of "
            "the clip); dialogue_ratio (fraction of runtime with spoken dialogue, 0-1); pacing_wpm (spoken "
            "words per minute); motion_intensity (0 locked-off, 1 continuously kinetic); sentiment (snake_case "
            "framing label such as stagnant_two_shot, kinetic_suspense, tense_dialogue, high_octane, "
            "ambient_build); action_level (0-1 physical action on screen); camera_motion (low, medium or "
            "high); music_intensity (0-1 prominence of score or music); emotional_intensity (0-1 charge of the "
            "performances); avg_shot_length_sec (estimate from the cutting rhythm); visible_characters "
            "(typical count on screen); summary (one or two plain sentences describing what happens)."
        )
        response, _ = generate_with_fallback(
            client,
            [video_part, prompt],
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SceneFeatures,
                temperature=0.1,
                max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_EXTRACT", "900")),
            ),
            label="extract",
        )
        return SceneFeatures.model_validate_json(response.text)
    except Exception as exc:  # noqa: BLE001
        print(f"[GEMINI EXTRACTION WARNING] Falling back to cut_01 preset: {exc}")
        return extract_scene_dna_mock("cut_01")


# --------------------------------------------------------------------------------------
# Stage 2: ClickHouse via the official MCP server
# --------------------------------------------------------------------------------------
class MCPClickHouseClient:
    """Keeps one `mcp-clickhouse` stdio server alive on a private event-loop thread.

    Every query the agent runs is an MCP `run_query` tool call against that server, which
    in turn talks to ClickHouse Cloud using the CLICKHOUSE_* environment variables.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self.server_info: str = ""
        self.tools: list[str] = []
        self.last_error: str = ""
        self.started_at: float = 0.0

    # -- lifecycle -------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._session is not None

    def start(self, timeout: float = 90.0) -> bool:
        with self._lock:
            if self.connected:
                return True
            self._ready.clear()
            self.last_error = ""
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name="mcp-clickhouse-loop", daemon=True
            )
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        self._ready.wait(timeout)
        if not self.connected:
            self.last_error = self.last_error or "timed out waiting for mcp-clickhouse to start"
            print(f"[MCP] failed to connect: {self.last_error}")
        return self.connected

    async def _run(self) -> None:
        env = dict(os.environ)
        env.setdefault("CLICKHOUSE_MCP_SERVER_TRANSPORT", "stdio")
        env.setdefault("CLICKHOUSE_SECURE", "true")
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_clickhouse.main"], env=env
        )
        self._stop_event = asyncio.Event()
        try:
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    init = await session.initialize()
                    self.server_info = f"{init.server_info.name} {init.server_info.version}"
                    listed = await session.list_tools()
                    self.tools = [t.name for t in listed.tools]
                    self._session = session
                    self.started_at = time.time()
                    print(f"[MCP] connected to {self.server_info}; tools={self.tools}")
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._session = None
            self._ready.set()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=5)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._session = None

    def restart(self) -> bool:
        self.stop()
        return self.start()

    # -- tool calls ------------------------------------------------------------------
    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 120.0) -> Any:
        if not self.connected:
            if not self.start():
                raise RuntimeError(f"mcp-clickhouse unavailable: {self.last_error}")
        assert self._session is not None and self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        result = future.result(timeout)
        text = "".join(getattr(part, "text", "") for part in result.content)
        if getattr(result, "is_error", False):
            raise RuntimeError(text or f"MCP tool {name} failed")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def run_query(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        payload = self.call_tool("run_query", {"query": sql})
        if isinstance(payload, dict) and "columns" in payload:
            return list(payload["columns"]), [list(r) for r in payload.get("rows", [])]
        if isinstance(payload, dict) and "raw" in payload:
            raise RuntimeError(payload["raw"][:500])
        raise RuntimeError(f"Unexpected MCP result: {str(payload)[:300]}")

    def list_tables(self, database: str = "default") -> list[str]:
        payload = self.call_tool("list_tables", {"database": database, "include_detailed_columns": False})
        names: list[str] = []
        if isinstance(payload, dict):
            for tbl in payload.get("tables", []) or []:
                if isinstance(tbl, dict) and tbl.get("name"):
                    names.append(tbl["name"])
        elif isinstance(payload, list):
            for tbl in payload:
                if isinstance(tbl, dict) and tbl.get("name"):
                    names.append(tbl["name"])
        return names


mcp_clickhouse = MCPClickHouseClient()


def ensure_mcp() -> bool:
    return mcp_clickhouse.connected or mcp_clickhouse.start()


def mcp_status() -> dict[str, Any]:
    return {
        "connected": mcp_clickhouse.connected,
        "server": mcp_clickhouse.server_info,
        "transport": "stdio",
        "tools": mcp_clickhouse.tools,
        "host": os.getenv("CLICKHOUSE_HOST", ""),
        "last_error": mcp_clickhouse.last_error,
    }


# -- SQL templates (ClickHouse server-side parameter syntax) ---------------------------
SEGMENT_HAZARD_SQL = """
WITH global_baseline AS (
    SELECT sum(dropped_out) / count() AS global_rate
    FROM viewer_retention_events
),
matching_scenes AS (
    SELECT scene_id
    FROM scene_dna_features
    WHERE dialogue_ratio BETWEEN {dr_min:Float32} AND {dr_max:Float32}
      AND duration_sec BETWEEN {dur_min:UInt32} AND {dur_max:UInt32}
)
SELECT
    v.age_group AS age_group,
    v.device AS device,
    count() AS sample_size,
    sum(v.dropped_out) AS drops,
    round(sum(v.dropped_out) / count() * 100, 2) AS dropout_percentage,
    round((sum(v.dropped_out) / count()) / max(g.global_rate), 2) AS hazard_ratio
FROM viewer_retention_events AS v
CROSS JOIN global_baseline AS g
WHERE v.scene_id IN (SELECT scene_id FROM matching_scenes)
GROUP BY v.age_group, v.device
ORDER BY hazard_ratio DESC
""".strip()

DROP_CURVE_SQL = """
SELECT
    toUInt8(least(19, floor(v.timestamp_sec / s.duration_sec * 20))) AS bucket,
    count() AS views,
    sum(v.dropped_out) AS drops,
    round(sum(v.dropped_out) / count() * 100, 2) AS drop_pct
FROM viewer_retention_events AS v
INNER JOIN scene_dna_features AS s ON v.scene_id = s.scene_id
WHERE v.age_group = {age_group:String}
  AND v.device = {device:String}
  AND s.dialogue_ratio BETWEEN {dr_min:Float32} AND {dr_max:Float32}
  AND s.duration_sec BETWEEN {dur_min:UInt32} AND {dur_max:UInt32}
GROUP BY bucket
ORDER BY bucket
""".strip()

TOTAL_ROWS_SQL = "SELECT count() AS total FROM viewer_retention_events"

PATTERN_SQL = """
WITH (SELECT avg(dropped_out) FROM viewer_retention_events) AS global_rate
SELECT
    pattern,
    count() AS sample_size,
    round(avg(dropped_out) * 100, 2) AS dropout_percentage,
    round(avg(dropped_out) / global_rate, 2) AS hazard_ratio
FROM (
    SELECT
        v.dropped_out AS dropped_out,
        arrayJoin(arrayFilter(x -> x != '', [
            if(s.dialogue_ratio > 0.70 AND s.duration_sec > 240, 'Dialogue-heavy scenes over 4 min', ''),
            if(s.motion_intensity < 0.30, 'Static camera with minimal action', ''),
            if(s.pacing_wpm > 180, 'Fast dialogue pacing above 180 wpm', ''),
            if(s.duration_sec > 240, 'Runtime longer than 4 minutes', ''),
            if(s.dialogue_ratio BETWEEN 0.40 AND 0.70, 'Balanced dialogue and action', ''),
            if(s.motion_intensity > 0.60, 'Kinetic action coverage', '')
        ])) AS pattern
    FROM viewer_retention_events AS v
    INNER JOIN scene_dna_features AS s ON v.scene_id = s.scene_id
    WHERE v.age_group = {age_group:String} AND v.device = {device:String}
)
GROUP BY pattern
ORDER BY hazard_ratio DESC
""".strip()

SCATTER_SQL = """
SELECT
    s.scene_id AS scene_id,
    s.scene_title AS scene_title,
    s.duration_sec AS duration_sec,
    s.dialogue_ratio AS dialogue_ratio,
    v.age_group AS age_group,
    v.device AS device,
    count() AS views,
    round(avg(v.dropped_out) * 100, 2) AS exit_rate
FROM viewer_retention_events AS v
INNER JOIN scene_dna_features AS s ON v.scene_id = s.scene_id
GROUP BY scene_id, scene_title, duration_sec, dialogue_ratio, age_group, device
ORDER BY duration_sec
""".strip()

SCENE_TYPE_SQL = """
SELECT
    s.sentiment AS scene_type,
    count() AS views,
    sum(v.dropped_out) AS drops,
    round(avg(v.dropped_out) * 100, 2) AS exit_rate
FROM viewer_retention_events AS v
INNER JOIN scene_dna_features AS s ON v.scene_id = s.scene_id
GROUP BY scene_type
ORDER BY drops DESC
""".strip()

SCENE_COUNT_SQL = "SELECT count() AS scenes FROM scene_dna_features"

_READ_ONLY_RE = re.compile(r"^\s*(SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN)\b", re.IGNORECASE)


_PLACEHOLDER_RE = re.compile(r"\{(\w+):([A-Za-z0-9()]+)\}")


def bind_sql(sql: str, parameters: dict[str, Any]) -> str:
    """Render ClickHouse `{name:Type}` placeholders client-side so the exact SQL can be shipped
    over the MCP `run_query` tool (which takes a single SQL string) and shown in the evidence
    drawer. Numbers are emitted as literals; strings are quoted with ClickHouse escaping."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in parameters:
            raise KeyError(f"missing SQL parameter: {name}")
        value = parameters[name]
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return repr(value)
        escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    return _PLACEHOLDER_RE.sub(replace, sql)


def _thresholds(features: SceneFeatures) -> dict[str, Any]:
    """Profile band: scenes whose dialogue ratio is within +/-0.15 and whose runtime is within
    +/-30% of the analysed cut are treated as sharing its DNA."""
    return {
        "dr_min": float(round(max(0.0, features.dialogue_ratio - 0.15), 2)),
        "dr_max": float(round(min(1.0, features.dialogue_ratio + 0.15), 2)),
        "dur_min": int(features.duration_sec * 0.70),
        "dur_max": int(features.duration_sec * 1.30),
    }


class AgentTrace:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.model_used: str = ""

    def record(self, tool: str, status: str, latency_ms: float, detail: str, rows: int | None = None,
               actor: str = "pipeline") -> None:
        self.steps.append({
            "step": len(self.steps) + 1,
            "actor": actor,
            "tool": tool,
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "rows": rows,
            "detail": detail,
        })

    def timed_query(self, label: str, sql: str, actor: str = "pipeline") -> tuple[list[str], list[list[Any]], float]:
        started = time.perf_counter()
        try:
            columns, rows = mcp_clickhouse.run_query(sql)
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - started) * 1000
            self.record("mcp-clickhouse.run_query", "error", latency, f"{label}: {str(exc)[:200]}", actor=actor)
            raise
        latency = (time.perf_counter() - started) * 1000
        self.record("mcp-clickhouse.run_query", "ok", latency, label, rows=len(rows), actor=actor)
        return columns, rows, latency


def _fallback_evidence(features: SceneFeatures, params: dict[str, Any], reason: str, trace: AgentTrace) -> dict[str, Any]:
    segments = [
        {"age_group": "18-24", "device": "mobile", "sample_size": 119965, "drops": 28221, "dropout_percentage": 23.52, "hazard_ratio": 2.81},
        {"age_group": "45-54", "device": "tv", "sample_size": 119922, "drops": 9759, "dropout_percentage": 8.14, "hazard_ratio": 0.97},
        {"age_group": "55+", "device": "tablet", "sample_size": 120155, "drops": 9763, "dropout_percentage": 8.13, "hazard_ratio": 0.97},
        {"age_group": "18-24", "device": "tablet", "sample_size": 120186, "drops": 9777, "dropout_percentage": 8.13, "hazard_ratio": 0.97},
        {"age_group": "25-34", "device": "desktop", "sample_size": 119423, "drops": 9579, "dropout_percentage": 8.02, "hazard_ratio": 0.96},
        {"age_group": "35-44", "device": "mobile", "sample_size": 119876, "drops": 9590, "dropout_percentage": 8.00, "hazard_ratio": 0.96},
    ]
    curve = []
    for bucket in range(20):
        rel = bucket / 20
        peak = 23.5 if 0.45 <= rel <= 0.72 else 8.1
        curve.append({"bucket": bucket, "rel_start": rel, "rel_end": rel + 0.05, "views": 6000, "drops": int(6000 * peak / 100), "drop_pct": peak})
    return {
        "transport": "offline-fallback",
        "mcp_server": mcp_clickhouse.server_info or "mcp-clickhouse (not connected)",
        "fallback": True,
        "fallback_reason": reason,
        "latency_ms": 0.0,
        "query_latency_ms": {"segments": 0.0, "curve": 0.0, "count": 0.0},
        "rows_scanned": TOTAL_EVENTS_EXPECTED,
        "sql": SEGMENT_HAZARD_SQL,
        "bound_sql": bind_sql(SEGMENT_HAZARD_SQL, params),
        "curve_sql": DROP_CURVE_SQL,
        "parameters": params,
        "segments": segments,
        "top_risk": segments[0],
        "curve": curve,
        "patterns": [
            {"pattern": "Dialogue-heavy scenes over 4 min", "sample_size": 120000, "dropout_percentage": 23.5, "hazard_ratio": 2.7},
            {"pattern": "Static camera with minimal action", "sample_size": 120000, "dropout_percentage": 23.4, "hazard_ratio": 2.7},
            {"pattern": "Fast dialogue pacing above 180 wpm", "sample_size": 120000, "dropout_percentage": 23.3, "hazard_ratio": 2.6},
            {"pattern": "Runtime longer than 4 minutes", "sample_size": 120000, "dropout_percentage": 23.2, "hazard_ratio": 2.6},
            {"pattern": "Kinetic action coverage", "sample_size": 80000, "dropout_percentage": 8.0, "hazard_ratio": 0.9},
        ],
        "scatter": [],
        "scene_types": [
            {"scene_type": "stagnant_two_shot", "views": 800000, "drops": 100000, "exit_rate": 12.5},
            {"scene_type": "dramatic_speech", "views": 800000, "drops": 98000, "exit_rate": 12.3},
            {"scene_type": "tense_dialogue", "views": 800000, "drops": 96000, "exit_rate": 12.0},
            {"scene_type": "kinetic_suspense", "views": 800000, "drops": 64000, "exit_rate": 8.0},
            {"scene_type": "high_octane", "views": 800000, "drops": 64000, "exit_rate": 8.0},
            {"scene_type": "ambient_build", "views": 800000, "drops": 64000, "exit_rate": 8.0},
        ],
        "scene_count": 6,
        "tables": ["scene_dna_features", "viewer_retention_events"],
    }


def query_clickhouse_mcp(features: SceneFeatures, trace: AgentTrace | None = None) -> dict[str, Any]:
    """Run the OLAP evidence queries through the ClickHouse MCP server."""
    trace = trace or AgentTrace()
    params = _thresholds(features)

    if not ensure_mcp():
        trace.record("mcp-clickhouse.initialize", "error", 0.0, mcp_clickhouse.last_error[:200])
        return _fallback_evidence(features, params, mcp_clickhouse.last_error, trace)

    try:
        # Step A: schema discovery through the MCP `list_tables` tool.
        started = time.perf_counter()
        tables = mcp_clickhouse.list_tables("default")
        trace.record("mcp-clickhouse.list_tables", "ok", (time.perf_counter() - started) * 1000,
                     f"database=default -> {', '.join(tables) if tables else 'no tables'}", rows=len(tables))

        # Step B: demographic hazard aggregation across all matching scenes.
        bound_segments = bind_sql(SEGMENT_HAZARD_SQL, params)
        columns, rows, seg_ms = trace.timed_query(
            f"segment hazard ratios (dialogue_ratio {params['dr_min']}-{params['dr_max']}, duration {params['dur_min']}-{params['dur_max']}s)",
            bound_segments,
        )
        idx = {c: i for i, c in enumerate(columns)}
        segments = [{
            "age_group": str(r[idx["age_group"]]),
            "device": str(r[idx["device"]]),
            "sample_size": int(r[idx["sample_size"]]),
            "drops": int(r[idx["drops"]]),
            "dropout_percentage": float(r[idx["dropout_percentage"]]),
            "hazard_ratio": float(r[idx["hazard_ratio"]]),
        } for r in rows]
        top_risk = segments[0] if segments else None

        # Step C: where in the runtime does the top-risk cohort leave? (20 relative buckets)
        curve: list[dict[str, Any]] = []
        curve_ms = 0.0
        curve_params = dict(params)
        if top_risk:
            curve_params.update({"age_group": top_risk["age_group"], "device": top_risk["device"]})
            c_cols, c_rows, curve_ms = trace.timed_query(
                f"drop-off curve for {top_risk['age_group']} / {top_risk['device']} (20 runtime buckets)",
                bind_sql(DROP_CURVE_SQL, curve_params),
            )
            cidx = {c: i for i, c in enumerate(c_cols)}
            by_bucket = {int(r[cidx["bucket"]]): r for r in c_rows}
            for bucket in range(20):
                r = by_bucket.get(bucket)
                views = int(r[cidx["views"]]) if r else 0
                drops = int(r[cidx["drops"]]) if r else 0
                curve.append({
                    "bucket": bucket,
                    "rel_start": bucket / 20,
                    "rel_end": (bucket + 1) / 20,
                    "views": views,
                    "drops": drops,
                    "drop_pct": float(r[cidx["drop_pct"]]) if r else 0.0,
                })

        # Step D: total rows scanned for the evidence footer.
        _, count_rows, count_ms = trace.timed_query("total retention events in scope", TOTAL_ROWS_SQL)
        rows_scanned = int(count_rows[0][0]) if count_rows else TOTAL_EVENTS_EXPECTED

        # Step E: historical pattern discovery, scatter and scene-type mix for the dashboard.
        patterns: list[dict[str, Any]] = []
        scatter: list[dict[str, Any]] = []
        scene_types: list[dict[str, Any]] = []
        scene_count = 0
        if top_risk:
            p_cols, p_rows, _ = trace.timed_query(
                f"pattern discovery for {top_risk['age_group']} / {top_risk['device']}",
                bind_sql(PATTERN_SQL, {"age_group": top_risk["age_group"], "device": top_risk["device"]}),
            )
            pidx = {c: i for i, c in enumerate(p_cols)}
            patterns = [{
                "pattern": str(r[pidx["pattern"]]),
                "sample_size": int(r[pidx["sample_size"]]),
                "dropout_percentage": float(r[pidx["dropout_percentage"]]),
                "hazard_ratio": float(r[pidx["hazard_ratio"]]),
            } for r in p_rows]
        sc_cols, sc_rows, _ = trace.timed_query("scene duration vs exit rate (all cohorts)", SCATTER_SQL)
        sidx = {c: i for i, c in enumerate(sc_cols)}
        scatter = [{
            "scene_id": str(r[sidx["scene_id"]]),
            "scene_title": str(r[sidx["scene_title"]]),
            "duration_sec": int(r[sidx["duration_sec"]]),
            "dialogue_ratio": float(r[sidx["dialogue_ratio"]]),
            "age_group": str(r[sidx["age_group"]]),
            "device": str(r[sidx["device"]]),
            "views": int(r[sidx["views"]]),
            "exit_rate": float(r[sidx["exit_rate"]]),
        } for r in sc_rows]
        st_cols, st_rows, _ = trace.timed_query("drop-off share by scene type", SCENE_TYPE_SQL)
        tidx = {c: i for i, c in enumerate(st_cols)}
        scene_types = [{
            "scene_type": str(r[tidx["scene_type"]]),
            "views": int(r[tidx["views"]]),
            "drops": int(r[tidx["drops"]]),
            "exit_rate": float(r[tidx["exit_rate"]]),
        } for r in st_rows]
        _, sc_count_rows, _ = trace.timed_query("scenes in library", SCENE_COUNT_SQL)
        scene_count = int(sc_count_rows[0][0]) if sc_count_rows else 0

        return {
            "transport": "mcp-clickhouse (stdio)",
            "mcp_server": mcp_clickhouse.server_info,
            "fallback": False,
            "latency_ms": round(seg_ms, 1),
            "query_latency_ms": {"segments": round(seg_ms, 1), "curve": round(curve_ms, 1), "count": round(count_ms, 1)},
            "rows_scanned": rows_scanned,
            "sql": SEGMENT_HAZARD_SQL,
            "bound_sql": bound_segments,
            "curve_sql": DROP_CURVE_SQL,
            "parameters": params,
            "segments": segments,
            "top_risk": top_risk,
            "curve": curve,
            "patterns": patterns,
            "scatter": scatter,
            "scene_types": scene_types,
            "scene_count": scene_count,
            "tables": tables,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[MCP WARNING] evidence query failed, using offline fallback: {exc}")
        mcp_clickhouse.last_error = str(exc)
        return _fallback_evidence(features, params, str(exc)[:200], trace)


# --------------------------------------------------------------------------------------
# Stage 3: Gemini synthesis with a ClickHouse tool in hand
# --------------------------------------------------------------------------------------
_CLICKHOUSE_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="clickhouse_run_query",
        description=(
            "Run a read-only ClickHouse SQL query (SELECT/WITH only) through the mcp-clickhouse "
            "server against the retention warehouse. Tables: viewer_retention_events(event_id UUID, "
            "viewer_id UInt32, episode_id String, scene_id String, age_group String ['18-24','25-34',"
            "'35-44','45-54','55+'], device String ['mobile','desktop','tv','tablet'], dropped_out UInt8, "
            "timestamp_sec UInt32) and scene_dna_features(scene_id, episode_id, scene_title, duration_sec, "
            "dialogue_ratio, pacing_wpm, motion_intensity, sentiment). Always aggregate; never return raw rows. "
            "Returns JSON with columns and rows."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"query": types.Schema(type=types.Type.STRING, description="ClickHouse SQL")},
            required=["query"],
        ),
    )
])


def _evidence_digest(features: SceneFeatures, ch: dict[str, Any]) -> str:
    top = ch.get("top_risk") or {}
    seg_lines = "\n".join(
        f"  - {s['age_group']:>5} / {s['device']:<7} n={s['sample_size']:,} drops={s['drops']:,} "
        f"rate={s['dropout_percentage']:.2f}% hazard={s['hazard_ratio']:.2f}x"
        for s in ch.get("segments", [])[:8]
    )
    curve = ch.get("curve", [])
    curve_lines = ""
    if curve:
        dur = features.duration_sec
        curve_lines = "\n".join(
            f"  - {format_timecode(b['rel_start'] * dur)}-{format_timecode(b['rel_end'] * dur)}: "
            f"{b['drop_pct']:.1f}% drop ({b['drops']:,}/{b['views']:,})"
            for b in curve
        )
    return f"""SCENE DNA
- Title: {features.scene_title}
- Summary: {features.summary or 'n/a'}
- Duration: {features.duration_sec}s ({format_timecode(features.duration_sec)})
- Dialogue ratio: {features.dialogue_ratio:.2f}
- Pacing: {features.pacing_wpm} WPM
- Motion intensity: {features.motion_intensity:.2f} (camera motion {features.camera_motion})
- Action level: {features.action_level:.2f}
- Music intensity: {features.music_intensity:.2f}
- Emotional intensity: {features.emotional_intensity:.2f}
- Avg shot length: {features.avg_shot_length_sec:.1f}s, visible characters: {features.visible_characters}
- Framing / sentiment: {features.sentiment}

CLICKHOUSE EVIDENCE (via {ch.get('transport')}, {ch.get('rows_scanned', 0):,} events, segment query {ch.get('latency_ms')} ms)
- Scene profile band: dialogue_ratio {ch.get('parameters', {}).get('dr_min')}-{ch.get('parameters', {}).get('dr_max')}, duration_sec {ch.get('parameters', {}).get('dur_min')}-{ch.get('parameters', {}).get('dur_max')}
- Top at-risk cohort: {top.get('age_group')} on {top.get('device')} -> {top.get('dropout_percentage')}% dropout, hazard {top.get('hazard_ratio')}x vs global baseline
- Segment table:
{seg_lines}
- Drop-off curve for the top cohort mapped onto this cut's runtime:
{curve_lines or '  (no curve available)'}
"""


def _run_gemini_drilldown(client: genai.Client, digest: str, trace: AgentTrace) -> str:
    """Let Gemini interrogate ClickHouse through the MCP-backed tool before prescribing."""
    if not AGENT_DRILLDOWN or not mcp_clickhouse.connected:
        return ""
    system = (
        "You are the SceneDNA retention agent. You may call clickhouse_run_query to verify or sharpen "
        "the evidence (for example: compare the risk cohort against the same cohort on low-dialogue "
        "scenes, or check whether the hazard holds per episode). Use at most "
        f"{AGENT_MAX_TOOL_CALLS} queries, aggregate only, then reply with 3-5 bullet findings and no SQL."
    )
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=digest)])]
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[_CLICKHOUSE_TOOL],
        temperature=0.2,
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_DRILLDOWN", "700")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    findings = ""
    calls = 0
    for _ in range(AGENT_MAX_TOOL_CALLS + 1):
        try:
            response, _ = generate_with_fallback(client, contents, config, trace, label="drilldown",
                                                 chain=_model_chain()[:2])
        except Exception as exc:  # noqa: BLE001
            # Drill-down is optional: skip it when Gemini is overloaded rather than stalling.
            trace.record("gemini.drilldown", "skipped", 0.0, f"skipped: {str(exc)[:120]}", actor="gemini")
            break
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            break
        contents.append(candidate.content)
        fn_calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
        if not fn_calls or calls >= AGENT_MAX_TOOL_CALLS:
            findings = " ".join(p.text for p in (candidate.content.parts or []) if getattr(p, "text", None)).strip()
            break
        response_parts: list[types.Part] = []
        for fc in fn_calls:
            calls += 1
            sql = str((fc.args or {}).get("query", "")).strip()
            started = time.perf_counter()
            if not _READ_ONLY_RE.match(sql):
                payload: dict[str, Any] = {"error": "Only read-only SELECT/WITH queries are permitted."}
                trace.record("mcp-clickhouse.run_query", "rejected", 0.0, f"gemini: non read-only SQL rejected", actor="gemini")
            else:
                try:
                    columns, rows = mcp_clickhouse.run_query(sql)
                    payload = {"columns": columns, "rows": rows[:50]}
                    trace.record("mcp-clickhouse.run_query", "ok", (time.perf_counter() - started) * 1000,
                                 f"gemini drill-down: {sql[:160]}", rows=len(rows), actor="gemini")
                except Exception as exc:  # noqa: BLE001
                    payload = {"error": str(exc)[:300]}
                    trace.record("mcp-clickhouse.run_query", "error", (time.perf_counter() - started) * 1000,
                                 f"gemini drill-down failed: {str(exc)[:160]}", actor="gemini")
            response_parts.append(types.Part.from_function_response(name=fc.name, response=payload))
        contents.append(types.Content(role="user", parts=response_parts))
    return findings


def _fallback_prescription(features: SceneFeatures, ch: dict[str, Any]) -> Prescription:
    top = ch.get("top_risk") or {"age_group": "18-24", "device": "mobile", "dropout_percentage": 23.5, "hazard_ratio": 2.9}
    dur = features.duration_sec
    if features.dialogue_ratio >= 0.6 and dur >= 200:
        start, end = int(dur * 0.50), int(dur * 0.65)
        cuts = [
            CutRecommendation(start_sec=start, end_sec=end, action="trim",
                              rationale=f"Remove {end - start}s of repeated exposition in the static two-shot where the {top['age_group']} {top['device']} cohort peaks in drop-off."),
            CutRecommendation(start_sec=int(dur * 0.38), end_sec=int(dur * 0.42), action="insert_cutaway",
                              rationale="Break the framing lock with a reaction insert before the risk window begins."),
        ]
        predicted = round(float(top.get("dropout_percentage", 23.5)) * 0.45, 1)
        headline = f"Trim the mid-scene dialogue lock: {format_timecode(start)}-{format_timecode(end)}"
        camera = [
            "Alternate the static two-shot with tight over-the-shoulder coverage every 12-15 seconds.",
            "Add a slow push-in on the speaker through the second half to restore visual momentum.",
            "Deliver a 9:16 punch-in variant for mobile so faces fill the frame.",
        ]
    else:
        cuts = []
        predicted = round(float(top.get("dropout_percentage", 8.0)), 1)
        headline = "Pacing is within tolerance. Lock the cut."
        camera = ["Keep the current kinetic coverage. No structural changes required."]
    cut_sentences = " ".join(
        f"{c.action.replace('_', ' ').capitalize()} between {format_timecode(c.start_sec)} and {format_timecode(c.end_sec)}: {c.rationale}"
        for c in cuts
    ) or "No structural cuts are required."
    report = (
        f"{headline}. The ClickHouse evidence shows scenes with this profile are associated with {top['age_group']} viewers on {top['device']} exiting at "
        f"{top['dropout_percentage']} percent, which is {top['hazard_ratio']} times the global baseline. "
        f"{cut_sentences} "
        f"Camera direction: {' '.join(camera)} "
        f"Predicted drop-off for the risk cohort after the edit is about {predicted} percent."
    )
    return Prescription(
        headline=headline,
        risk_summary=f"This scene profile is associated with {top['dropout_percentage']}% exits among {top['age_group']} {top['device']} viewers, {top['hazard_ratio']}x the baseline.",
        cuts=cuts,
        camera_directions=camera,
        predicted_drop_pct_after=predicted,
        confidence=0.6,
        report_text=report,
    )


def synthesize_prescription(features: SceneFeatures, ch_data: dict[str, Any], trace: AgentTrace | None = None) -> Prescription:
    trace = trace or AgentTrace()
    digest = _evidence_digest(features, ch_data)
    try:
        client = _gemini_client()

        started = time.perf_counter()
        findings = _run_gemini_drilldown(client, digest, trace)
        if findings:
            trace.record("gemini.drilldown", "ok", (time.perf_counter() - started) * 1000,
                         findings[:220], actor="gemini")

        prompt = f"""{digest}
AGENT DRILL-DOWN FINDINGS
{findings or '(none)'}

You are a senior film editor and retention analyst. Write a decisive editorial prescription for the edit suite.
Rules:
- The evidence is correlational. Never say the scene caused viewers to leave. Use phrasing such as "is associated with", "correlates with", "matches a historical risk pattern", "evidence suggests", "likely retention risk".
- Every cut must fall inside 0 and {features.duration_sec} seconds and use the drop-off curve to place it where the risk cohort actually leaves.
- Prefer 1-3 cuts. Use action values: trim, tighten, insert_cutaway, reframe, reorder.
- Give concrete camera direction (coverage, lens movement, mobile framing).
- predicted_drop_pct_after is the expected dropout % for the top cohort after the edit.
- report_text is a studio memo in plain prose: 3 to 5 short paragraphs, no markdown, no asterisks, no bullet symbols, no dashes as separators, no headings. Name timestamped cuts inline like 'trim 42 seconds between 02:15 and 02:57'. No preamble.
- If the scene profile is low-risk, say so and recommend locking the cut with no or minimal changes.
"""
        started = time.perf_counter()
        response, model_used = generate_with_fallback(
            client,
            prompt,
            types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Prescription,
                temperature=0.3,
                max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_SYNTH", "1800")),
            ),
            trace,
            label="synthesize",
        )
        prescription = Prescription.model_validate_json(response.text)
        trace.model_used = model_used
        # Clamp cut ranges into the scene runtime.
        for cut in prescription.cuts:
            cut.start_sec = max(0, min(cut.start_sec, features.duration_sec))
            cut.end_sec = max(cut.start_sec, min(cut.end_sec, features.duration_sec))
        trace.record("gemini.synthesize", "ok", (time.perf_counter() - started) * 1000,
                     prescription.headline[:200], actor="gemini")
        return prescription
    except Exception as exc:  # noqa: BLE001
        print(f"[GEMINI SYNTHESIS WARNING] Fallback prescription used: {exc}")
        trace.record("gemini.synthesize", "fallback", 0.0, str(exc)[:200], actor="gemini")
        return _fallback_prescription(features, ch_data)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
RISK_FACTORS = [
    ("Dialogue ratio above 75%", lambda f: f.dialogue_ratio > 0.75),
    ("Duration above 4 minutes", lambda f: f.duration_sec > 240),
    ("Low camera motion", lambda f: f.motion_intensity < 0.30 or f.camera_motion.lower() == "low"),
    ("Low music intensity", lambda f: f.music_intensity < 0.30),
]


def risk_scan(features: SceneFeatures, evidence: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
    """Risk Analyst: compare the new scene against the historically discovered risk factors."""
    started = time.perf_counter()
    factors = [{"label": label, "matched": bool(check(features))} for label, check in RISK_FACTORS]
    matched = sum(1 for f in factors if f["matched"])
    top = evidence.get("top_risk") or {}
    hazard = float(top.get("hazard_ratio", 0.0))
    if matched >= 3 and hazard >= 2.0:
        level = "HIGH"
    elif matched >= 2 or hazard >= 1.3:
        level = "MEDIUM"
    else:
        level = "LOW"
    statement = (
        f"Matches {matched}/{len(factors)} historical risk factors. Scenes with this profile are associated with "
        f"{top.get('dropout_percentage', 0)}% exits among {top.get('age_group', '?')} {top.get('device', '?')} viewers "
        f"({hazard:.2f}x baseline)."
    )
    trace.record("risk.scan", "ok", (time.perf_counter() - started) * 1000, statement, rows=matched, actor="risk_analyst")
    return {"level": level, "matched": matched, "total": len(factors), "factors": factors, "statement": statement}


def run_pipeline(features: SceneFeatures, scene_id: str, source: str) -> dict[str, Any]:
    trace = AgentTrace()
    started = time.perf_counter()
    evidence = query_clickhouse_mcp(features, trace)
    risk = risk_scan(features, evidence, trace)
    prescription = synthesize_prescription(features, evidence, trace)
    synth_ok = any(t["tool"] == "gemini.synthesize" and t["status"] == "ok" for t in trace.steps)
    if evidence.get("fallback"):
        mode = {"mode": "demo", "label": "Demo mode", "detail": "ClickHouse MCP unreachable; showing a deterministic scenario"}
    elif not synth_ok:
        mode = {"mode": "partial", "label": "Live data, AI fallback", "detail": "ClickHouse MCP live; Gemini temporarily unavailable so the recommendation is rule-based"}
    else:
        mode = {"mode": "live", "label": "Live mode", "detail": "Gemini + ClickHouse MCP active"}
    return {
        "mode": mode,
        "risk": risk,
        "scene_id": scene_id,
        "source": source,
        "features": features.model_dump(),
        "clickhouse": evidence,
        "prescription": prescription.report_text,
        "prescription_structured": prescription.model_dump(),
        "agent_trace": trace.steps,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": trace.model_used or GEMINI_MODEL,
        "model_chain": _model_chain(),
    }


# --------------------------------------------------------------------------------------
# Ask anything: Gemini answers natural-language questions by querying ClickHouse over MCP
# --------------------------------------------------------------------------------------
def ask_warehouse(question: str, max_calls: int = 3) -> dict[str, Any]:
    trace = AgentTrace()
    started = time.perf_counter()
    if not ensure_mcp():
        return {"question": question, "answer": f"ClickHouse MCP server unavailable: {mcp_clickhouse.last_error}",
                "queries": [], "agent_trace": trace.steps, "total_latency_ms": 0.0, "model": GEMINI_MODEL}
    client = _gemini_client()
    system = (
        "You are the SceneDNA audience-intelligence agent for a film studio. Answer the user's question about "
        "viewer retention by calling clickhouse_run_query (read-only, aggregate SQL only; always alias columns; "
        f"use at most {max_calls} queries). scene_id values look like cut_01..cut_06 and join to scene_dna_features. "
        "dropped_out is 1 when the viewer abandoned the scene. Then answer in 2-4 plain sentences with concrete "
        "numbers. No markdown, no bullet symbols, no asterisks."
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    config = types.GenerateContentConfig(
        system_instruction=system, tools=[_CLICKHOUSE_TOOL], temperature=0.2,
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_ASK", "600")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    answer = ""
    queries: list[dict[str, Any]] = []
    calls = 0
    model_used = GEMINI_MODEL
    for _ in range(max_calls + 1):
        response, model_used = generate_with_fallback(client, contents, config, trace, label="ask")
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            break
        contents.append(candidate.content)
        fn_calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
        if not fn_calls or calls >= max_calls:
            answer = " ".join(p.text for p in (candidate.content.parts or []) if getattr(p, "text", None)).strip()
            break
        parts: list[types.Part] = []
        for fc in fn_calls:
            calls += 1
            sql = str((fc.args or {}).get("query", "")).strip()
            t0 = time.perf_counter()
            if not _READ_ONLY_RE.match(sql):
                payload: dict[str, Any] = {"error": "Only read-only SELECT/WITH queries are permitted."}
                trace.record("mcp-clickhouse.run_query", "rejected", 0.0, "non read-only SQL rejected", actor="gemini")
            else:
                try:
                    columns, rows = mcp_clickhouse.run_query(sql)
                    ms = (time.perf_counter() - t0) * 1000
                    payload = {"columns": columns, "rows": rows[:50]}
                    queries.append({"sql": sql, "columns": columns, "rows": rows[:20], "latency_ms": round(ms, 1)})
                    trace.record("mcp-clickhouse.run_query", "ok", ms, sql[:160], rows=len(rows), actor="gemini")
                except Exception as exc:  # noqa: BLE001
                    payload = {"error": str(exc)[:300]}
                    trace.record("mcp-clickhouse.run_query", "error", (time.perf_counter() - t0) * 1000, str(exc)[:160], actor="gemini")
            parts.append(types.Part.from_function_response(name=fc.name, response=payload))
        contents.append(types.Content(role="user", parts=parts))
    if not answer:
        answer = "I could not form an answer from the warehouse this time. Try rephrasing the question."
    return {
        "question": question,
        "answer": answer,
        "queries": queries,
        "agent_trace": trace.steps,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": model_used,
    }
