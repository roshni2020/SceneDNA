import os
import json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

prompt = """
You are an expert full-stack engineer. Generate the complete source code for SceneDNA:
1. requirements.txt (fastapi, uvicorn, clickhouse-connect, google-genai, etc.)
2. seed_clickhouse.py (vectorized script seeding 4.8M synthetic retention events)
3. agent_pipeline.py (Gemini multimodal feature extractor + ClickHouse query with latency measurement)
4. main.py (FastAPI backend with endpoints /api/analyze-preset and static frontend serving)
5. frontend/package.json, frontend/vite.config.ts, frontend/index.html, frontend/src/App.tsx, frontend/src/index.css
6. Dockerfile (multi-stage node:20 + python:3.11 for Cloud Run)

Output ONLY a valid JSON object where keys are relative file paths and values are strings of the complete file content. No placeholders or truncation.
"""

MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]

print("Gemini is generating the entire project architecture...")
response = None
for model in MODELS:
    try:
        print(f"Trying {model}...")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=65536,
            ),
        )
        break
    except Exception as e:  # noqa: BLE001
        print(f"  {model} failed: {str(e)[:160]}")
if response is None:
    raise SystemExit("All models failed.")

files = json.loads(response.text)

for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path}")

print("\nProject scaffolded successfully.")
