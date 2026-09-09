# SceneDNA

Pre-release video intelligence for editors. SceneDNA reads a scene cut with Gemini, interrogates
4.8 million historical viewer retention events in ClickHouse Cloud through the official ClickHouse
MCP server, and returns a timestamped editorial prescription before the cut ships.

Built for the Agentic Cinema: The Blockbuster Hackathon, ClickHouse track.

## What it does

1. **Scene DNA extraction.** Gemini watches an unreleased cut and returns structured creative
   attributes: dialogue ratio, pacing in words per minute, motion intensity, framing sentiment.
2. **OLAP evidence at runtime.** The agent runs parameterized analytical SQL against
   `viewer_retention_events` (4.8M rows) and `scene_dna_features` through `mcp-clickhouse`,
   the official ClickHouse MCP server. It computes drop-out rate and hazard ratio per age group
   and device for scenes sharing the cut's profile, plus a drop-off curve across the runtime.
   Every tool call is recorded in an agent trace with latency and row counts.
3. **Agentic drill-down.** Gemini is handed a `clickhouse_run_query` tool routed through the same
   MCP session and may run its own verification queries before prescribing.
4. **Editorial prescription.** Gemini synthesizes the evidence into a structured directive: exact
   cut ranges, camera direction, and the predicted drop-off after the edit.
5. **Studio UI.** A monitor viewport, a retention risk scrubber colored by the real drop-off curve
   with the recommended cuts overlaid, the directive, and an evidence drawer showing latency,
   rows scanned, the exact SQL sent over MCP, the agent tool trace, and the demographic table.

Example finding on the seeded warehouse: 18-24 viewers on mobile abandon long, dialogue-locked
scenes at 23.5 percent, about 2.8 times the global baseline.

## Architecture

```
Browser (Vite + React + Tailwind)
        |
        v
FastAPI  main.py
        |-- agent_pipeline.py
        |     |-- Gemini (google-genai): feature extraction, drill-down tool calls, synthesis
        |     |-- MCPClickHouseClient: persistent stdio session to `mcp-clickhouse`
        |             |-- tools: list_tables, run_query
        |             v
        |        ClickHouse Cloud: viewer_retention_events (4.8M), scene_dna_features
        v
Single container on Cloud Run (frontend built into the image)
```

The data in `viewer_retention_events` is synthetic. `seed_clickhouse.py` generates it in vectorized
NumPy chunks with a deterministic correlation: dialogue ratio above 0.70 and duration above 240 s
produce a 23.5 percent drop rate for the 18-24 mobile cohort against an 8 percent baseline, with
drop-offs clustered in the mid-scene window.

## Run locally

Requirements: Python 3.12+, Node 20+, a ClickHouse Cloud service, a Gemini API key.

```bash
cp .env.example .env            # fill in CLICKHOUSE_* and GEMINI_API_KEY
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt
python seed_clickhouse.py       # loads 6 scenes + 4.8M events (resumable; --reset to regenerate)

cd frontend && npm install && npm run build && cd ..
uvicorn main:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080.

For frontend development with hot reload, run `npm run dev` inside `frontend/`. The dev server
proxies `/api` to port 8080.

## API

| Method | Path                  | Purpose                                                        |
| ------ | --------------------- | -------------------------------------------------------------- |
| POST   | `/api/analyze-preset` | `{ "scene_id": "cut_01" \| "cut_02" }` runs the full pipeline |
| POST   | `/api/analyze-video`  | multipart upload; Gemini extracts DNA, then the same pipeline  |
| GET    | `/api/health`         | ClickHouse, MCP server, and Gemini readiness                   |
| GET    | `/api/mcp/status`     | MCP session state and registered tools                         |
| POST   | `/api/mcp/reconnect`  | restart the MCP server process                                 |

The analysis response includes `features`, `clickhouse` (segments, curve, SQL, latency, rows),
`prescription_structured`, `prescription` (plain-prose memo), and `agent_trace`.

## Deploy to Cloud Run

```bash
gcloud run deploy scenedna \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --set-env-vars CLICKHOUSE_HOST=...,CLICKHOUSE_PORT=8443,CLICKHOUSE_USER=default,CLICKHOUSE_SECURE=true,GEMINI_MODEL=gemini-3.5-flash \
  --set-secrets CLICKHOUSE_PASSWORD=clickhouse-password:latest,GEMINI_API_KEY=gemini-api-key:latest
```

Create the two secrets first with `gcloud secrets create`. The Dockerfile builds the frontend in a
Node stage and serves it from the Python image.

## Environment variables

| Variable                 | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| `CLICKHOUSE_HOST`        | ClickHouse Cloud host name                               |
| `CLICKHOUSE_PORT`        | HTTPS port, usually 8443                                 |
| `CLICKHOUSE_USER`        | Database user                                            |
| `CLICKHOUSE_PASSWORD`    | Database password                                        |
| `CLICKHOUSE_SECURE`      | `true` for ClickHouse Cloud                              |
| `GEMINI_API_KEY`         | Google AI Studio key                                     |
| `GEMINI_MODEL`           | Primary model, default `gemini-3.5-flash`                |
| `GEMINI_FALLBACK_MODELS` | Comma list tried on overload                             |
| `AGENT_MAX_TOOL_CALLS`   | Max Gemini drill-down queries per analysis, default 2    |
| `PORT`                   | HTTP port, default 8080                                  |

## Gemini through Google Cloud (Vertex AI)

Set `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` and the
app serves Gemini through your Google Cloud project with Application Default Credentials instead of
an AI Studio key. On Cloud Run this is the service account; locally run `gcloud auth application-default login`.
Enable the Vertex AI API on the project first.

## Data sources

- `viewer_retention_events`: 4.8 million synthetic viewer events (viewer, scene, age group, device,
  dropped_out flag, timestamp) generated by `seed_clickhouse.py` and stored in ClickHouse Cloud.
  No real audience data is used. The correlation between long dialogue-heavy scenes and 18-24
  mobile drop-off is programmed into the generator so the analytics have a signal to discover.
- `scene_dna_features`: six reference scenes with creative attributes, seeded by the same script.
- Uploaded video: analysed by Gemini at request time and never stored.

## Findings and learnings

- Routing every query through `mcp-clickhouse` costs about 100-400 ms per call on a warm stdio
  session, which is fast enough to run five aggregation queries per analysis and still feel live.
- Letting Gemini call ClickHouse itself produced useful verification queries (for example comparing
  the risk cohort on high-dialogue versus low-dialogue scenes) without hand-written prompts per case.
- Structured output for the prescription (typed cut ranges) made the UI far more useful than free
  text: cuts render directly on the timeline and in the recommendations list.
- Gemini model availability fluctuates; a fallback chain across models plus a short result cache kept
  the studio UI responsive during overload.

## License

MIT. See `LICENSE`.
