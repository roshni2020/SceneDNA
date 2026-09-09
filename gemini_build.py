"""Drive Gemini to author every SceneDNA file from the hackathon spec.

Gemini writes the code. This script only splits the spec into groups so no
single response is long enough to truncate, retries across models, and writes
the returned files to disk.
"""
import json
import os
import sys
import time

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

SYSTEM = """You are a Principal Full-Stack Engineer and Distributed Systems Architect specializing in Google Cloud (Vertex AI / Gemini) and ClickHouse OLAP databases.

We are building "SceneDNA" for the Google Cloud Agentic Cinema Hackathon (ClickHouse Track).
SceneDNA is a pre-release video intelligence platform:
1. Gemini extracts multimodal creative attributes (dialogue ratio, pacing WPM, motion intensity, framing, sentiment) from unreleased movie/TV scene cuts.
2. The agent executes high-speed analytical queries via ClickHouse against 4.8 million historical viewer retention records to identify statistically significant churn hazards across demographics (especially 18-24 mobile viewers).
3. Gemini synthesizes the OLAP findings into an actionable editorial prescription (specific timestamps to cut/trim, camera pacing adjustments).
4. A production-ready studio UI presents a video scrubber, risk heatmap, and an expandable "Inspect ClickHouse Evidence" drawer displaying millisecond latency, scanned rows, and the exact parameterized SQL query.

CRITICAL RULE: Do NOT use placeholders, truncation, ellipses (...), or "// implement here" comments. Write out every single line of code, imports, schemas, endpoints, and frontend components in full. Code must be production-ready and internally consistent across files.

SDK NOTES (must follow):
- Use the modern `google-genai` SDK: `from google import genai`, `from google.genai import types`, `client = genai.Client(api_key=...)`, `client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=PydanticModel))`.
- The Gemini API now rejects the model id "gemini-2.5-flash" for new API keys. Read the model id from the environment variable GEMINI_MODEL with default "gemini-3.5-flash". Never hardcode "gemini-2.5-flash".
- Use `clickhouse_connect.get_client(host=..., port=..., username=..., password=..., secure=...)`.
- ClickHouse parameterized queries use `client.query(sql, parameters={...})` with `{name:Type}` placeholders.

OUTPUT FORMAT: Respond with ONLY a valid JSON object. Keys are the relative file paths exactly as listed. Values are the complete file contents as strings. No markdown fences, no commentary.
"""

GROUPS = [
    {
        "name": "backend",
        "files": [
            "requirements.txt",
            ".env.example",
            "seed_clickhouse.py",
            "agent_pipeline.py",
            "main.py",
            "Dockerfile",
        ],
        "spec": """Generate these files: requirements.txt, .env.example, seed_clickhouse.py, agent_pipeline.py, main.py, Dockerfile.

#### requirements.txt
Include exact pinned versions for: fastapi, uvicorn[standard], google-genai (pin 2.22.0), clickhouse-connect, pydantic (v2), python-dotenv, numpy, pandas, python-multipart.

#### .env.example
Define environment variables: CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_SECURE, GEMINI_API_KEY, GEMINI_MODEL, PORT.

#### seed_clickhouse.py
- Connect to ClickHouse Cloud using clickhouse_connect with credentials from environment variables (load .env via python-dotenv).
- Create two MergeTree tables if they don't exist:
  - `scene_dna_features` (scene_id String, episode_id LowCardinality(String), scene_title String, duration_sec UInt32, dialogue_ratio Float32, pacing_wpm UInt16, motion_intensity Float32, sentiment LowCardinality(String)) ORDER BY (episode_id, scene_id).
  - `viewer_retention_events` (event_id UUID, viewer_id UInt32, episode_id LowCardinality(String), scene_id LowCardinality(String), age_group LowCardinality(String), device LowCardinality(String), dropped_out UInt8, timestamp_sec UInt32) ORDER BY (episode_id, age_group, device, scene_id).
- Seed a baseline library of 6 diverse scenes into `scene_dna_features`.
- Fast-generate 4,800,000 synthetic viewer drop-off events in vectorized NumPy chunks (800k rows/chunk, 6 chunks) and stream them using `client.insert_df()` with a pandas DataFrame. Generate UUIDs efficiently (e.g. via uuid4 list comprehension or numpy-based hex).
- Program deterministic correlation: scenes with `dialogue_ratio > 0.70` and `duration_sec > 240` must produce a ~23.5% drop rate among '18-24' age group on 'mobile' devices (versus a baseline ~8% drop rate across other segments), creating a ~2.9x hazard ratio.
- Age groups: '18-24', '25-34', '35-44', '45-54', '55+'. Devices: 'mobile', 'desktop', 'tv', 'tablet'.
- Print clean terminal progress with chunk timers and total elapsed time.

#### agent_pipeline.py
- Load .env via python-dotenv. Define a Pydantic schema `SceneFeatures` with fields: `scene_title: str`, `duration_sec: int`, `dialogue_ratio: float`, `pacing_wpm: int`, `motion_intensity: float`, `sentiment: str`.
- Implement `extract_scene_dna_mock(scene_id: str) -> SceneFeatures` with preset cuts:
  - "cut_01": "Safehouse Dialogue Lock (Cut 3)" (272s, 0.84 dialogue ratio, 205 wpm, 0.14 motion, "stagnant_two_shot").
  - "cut_02": "Warehouse Breach Pre-Assault" (135s, 0.22 dialogue ratio, 75 wpm, 0.76 motion, "kinetic_suspense").
- Implement `extract_scene_dna_from_video(video_bytes: bytes, mime_type: str) -> SceneFeatures` using the google-genai SDK with `types.Part.from_bytes(data=video_bytes, mime_type=mime_type)`, structured JSON schema output (response_schema=SceneFeatures), and a fallback to cut_01 preset on any exception.
- Implement `query_clickhouse_mcp(features: SceneFeatures) -> dict` that executes a parameterized SQL query on ClickHouse aggregating `sample_size` (count()), `drops` (sum(dropped_out)), `dropout_percentage`, and `hazard_ratio` (segment drop rate divided by the global drop rate across all rows, computed in SQL via a subquery or window) grouped by `age_group` and `device`, ordered by hazard_ratio DESC. Filter on scenes matching the feature profile (join or subquery against scene_dna_features using thresholds derived from the features, e.g. dialogue_ratio > {dr_threshold:Float32} AND duration_sec > {dur_threshold:UInt32}). Measure exact elapsed execution time via `time.perf_counter()` in milliseconds and count total rows scanned via `SELECT count() FROM viewer_retention_events` (expected 4.8M). Return a dict with keys: `latency_ms`, `rows_scanned`, `sql` (the exact parameterized SQL text), `parameters`, `segments` (list of dicts with age_group, device, sample_size, drops, dropout_percentage, hazard_ratio), and `top_risk` (the highest hazard segment). If ClickHouse is unreachable, return a clearly labelled fallback dict with representative numbers (18-24 mobile 23.5% / 2.94x, latency 38.4 ms, rows_scanned 4800000) and `fallback: true`.
- Implement `synthesize_prescription(features: SceneFeatures, ch_data: dict) -> str` using Gemini to generate a decisive studio report specifying exact timestamp cut ranges (e.g. "trim 42 seconds between 02:15 and 02:57") and camera direction advice. Include a deterministic fallback string if Gemini fails.
- Implement `format_timecode(seconds: int) -> str` returning MM:SS.

#### main.py
- FastAPI application with CORS middleware enabled for all origins.
- Pydantic request model `PresetRequest(scene_id: str)`.
- Endpoint `POST /api/analyze-preset`: accepts `{ "scene_id": string }`, runs feature extraction, ClickHouse OLAP query, and prescription synthesis, returning JSON `{scene_id, features, clickhouse, prescription}`.
- Endpoint `POST /api/analyze-video`: accepts multipart video upload (UploadFile), extracts multimodal features via Gemini, queries ClickHouse, and synthesizes prescription, same response shape.
- Endpoint `GET /api/health`: returns operational status with clickhouse and gemini readiness booleans.
- Mount `frontend/dist/assets` static files under `/assets` only if the directory exists, and provide a catch-all route `GET /{full_path:path}` serving `frontend/dist/index.html` for single-container deployment (returning a JSON 404 for paths starting with "api/").
- `if __name__ == "__main__":` run uvicorn on host 0.0.0.0 and PORT env (default 8080).

#### Dockerfile
Multi-stage build:
- Stage 1: `node:20-slim` builds the React frontend: copy frontend/package*.json, npm install, copy frontend, `npm run build`.
- Stage 2: `python:3.11-slim` installs dependencies from `requirements.txt`, copies backend files (main.py, agent_pipeline.py, seed_clickhouse.py), copies compiled frontend assets from stage 1 into `frontend/dist`, EXPOSE 8080, and starts `uvicorn main:app --host 0.0.0.0 --port 8080`.
""",
    },
    {
        "name": "frontend-config",
        "files": [
            "frontend/package.json",
            "frontend/tsconfig.json",
            "frontend/vite.config.ts",
            "frontend/tailwind.config.js",
            "frontend/postcss.config.js",
            "frontend/index.html",
            "frontend/src/main.tsx",
            "frontend/src/index.css",
        ],
        "spec": """Generate these files: frontend/package.json, frontend/tsconfig.json, frontend/vite.config.ts, frontend/tailwind.config.js, frontend/postcss.config.js, frontend/index.html, frontend/src/main.tsx, frontend/src/index.css.

Stack: Vite 5 + React 18 + TypeScript + Tailwind CSS 3 + lucide-react.

- frontend/package.json: name "scenedna-studio", scripts dev/build/preview, dependencies react, react-dom, lucide-react; devDependencies @types/react, @types/react-dom, @vitejs/plugin-react, typescript, vite, tailwindcss, autoprefixer, postcss. The build script must be just "vite build" (do not run tsc in build so the Docker build cannot fail on type nits).
- frontend/tsconfig.json: target ES2020, module ESNext, moduleResolution bundler, jsx react-jsx, strict true, include ["src"].
- frontend/vite.config.ts: react plugin, dev server proxy for "/api" to http://localhost:8080.
- frontend/tailwind.config.js: content ["./index.html", "./src/**/*.{ts,tsx}"], extend with a "pulse-risk" keyframe animation for red critical segments.
- frontend/postcss.config.js: tailwindcss + autoprefixer.
- frontend/index.html: title "SceneDNA Studio", dark background (#0a0a0f) on body, root div, module script to /src/main.tsx.
- frontend/src/main.tsx: React 18 createRoot rendering <App /> in StrictMode, importing ./index.css.
- frontend/src/index.css: Tailwind directives, dark body base styles, custom scrollbar, monospace code block styling.
""",
    },
    {
        "name": "app",
        "files": ["frontend/src/App.tsx"],
        "spec": """Generate this file: frontend/src/App.tsx (React 18 + TypeScript + Tailwind + lucide-react). It must be a single complete component file, fully typed, with no external state library.

Backend contract (already built): POST /api/analyze-preset with body {"scene_id": "cut_01" | "cut_02"} returns
{
  "scene_id": string,
  "features": { "scene_title": string, "duration_sec": number, "dialogue_ratio": number, "pacing_wpm": number, "motion_intensity": number, "sentiment": string },
  "clickhouse": { "latency_ms": number, "rows_scanned": number, "sql": string, "parameters": object, "segments": [{ "age_group": string, "device": string, "sample_size": number, "drops": number, "dropout_percentage": number, "hazard_ratio": number }], "top_risk": object, "fallback"?: boolean },
  "prescription": string
}
Define TypeScript interfaces for this response.

UI requirements:
- Studio header with active status indicators: "ClickHouse Cloud: 4.8M Events Connected" and "Gemini 1.5: Multimodal Active" (green dots).
- Video monitor viewport showing current scene title, live timecode (MM:SS), cut length, and animated play/pause states (a play button toggles a setInterval-driven playhead that advances one second per tick and loops at duration).
- Interactive Retention Risk Scrubber: color-coded segmented bar over the scene duration (emerald for safe pacing, amber for build, red pulse for critical churn risk starting at 02:15 (135s) for cut_01). Clicking a segment seeks the playhead. Show a moving playhead marker. For cut_02, all segments are emerald/amber with no critical zone.
- Quick-switch buttons for "Cut 01 - High Churn Risk Monologue" (scene_id cut_01) and "Cut 02 - Dynamic Action Cut" (scene_id cut_02); switching calls the API and shows a loading state.
- SceneDNA attributes card: dialogue percentage, pacing WPM, motion score, and sentiment tag.
- Studio Strategic Directive card displaying the prescriptive cut recommendation text from the API (pre-wrap).
- "INSPECT CLICKHOUSE EVIDENCE" collapsible drawer displaying:
  - Query latency in milliseconds formatted with one decimal (e.g. `38.4 ms`)
  - Total rows scanned formatted as `4.8M Rows`
  - The raw parameterized ClickHouse SQL query in a monospace code block
  - A demographic breakdown table showing Segment, Device, Drop Rate, and Hazard Ratio, highlighting rows where hazard_ratio >= 2 (e.g. 18-24 mobile at 23.5% / 2.94x) in red.
- Load cut_01 on mount. Handle fetch errors with a visible error banner. Use lucide-react icons (Play, Pause, Database, Cpu, ChevronDown, ChevronUp, Film, Activity, AlertTriangle, Scissors).
- Format helpers: formatTimecode(seconds), formatRows(n).
""",
    },
]


def call_gemini(prompt: str) -> dict:
    last_err = None
    for model in MODELS:
        for attempt in range(2):
            try:
                print(f"  -> {model} (attempt {attempt + 1})", flush=True)
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        response_mime_type="application/json",
                        max_output_tokens=65536,
                        temperature=0.2,
                    ),
                )
                return json.loads(resp.text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"     failed: {str(e)[:140]}", flush=True)
                time.sleep(2)
    raise SystemExit(f"All models failed: {last_err}")


def main() -> None:
    t0 = time.perf_counter()
    for group in GROUPS:
        print(f"\n== Generating group: {group['name']} ==", flush=True)
        files = call_gemini(group["spec"])
        missing = [f for f in group["files"] if f not in files]
        if missing:
            print(f"  WARNING missing from response: {missing}")
        for path, content in files.items():
            if not isinstance(content, str) or not content.strip():
                print(f"  WARNING empty content for {path}")
                continue
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content.rstrip("\n") + "\n")
            print(f"  Created: {path} ({len(content)} chars)")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
