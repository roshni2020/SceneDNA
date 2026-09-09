# Deploys SceneDNA to Cloud Run from source. Run from the project root in PowerShell after
# `gcloud auth login`. Reads ClickHouse and Gemini values from .env so nothing is typed twice.
param(
    [string]$Project = "project-eea71bf6-68cf-47c4-9ee",
    [string]$Region = "us-central1",
    [string]$Service = "scenedna"
)

$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { throw ".env not found next to deploy.ps1" }

$vars = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { $vars[$matches[1]] = $matches[2].Trim() }
}

gcloud config set project $Project | Out-Null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com

$plain = @(
    "CLICKHOUSE_HOST=$($vars['CLICKHOUSE_HOST'])",
    "CLICKHOUSE_PORT=$($vars['CLICKHOUSE_PORT'])",
    "CLICKHOUSE_USER=$($vars['CLICKHOUSE_USER'])",
    "CLICKHOUSE_PASSWORD=$($vars['CLICKHOUSE_PASSWORD'])",
    "CLICKHOUSE_SECURE=true",
    "GEMINI_API_KEY=$($vars['GEMINI_API_KEY'])",
    "GEMINI_MODEL=$($vars['GEMINI_MODEL'])",
    "GOOGLE_GENAI_USE_VERTEXAI=$($vars['GOOGLE_GENAI_USE_VERTEXAI'])",
    "GOOGLE_CLOUD_PROJECT=$Project",
    "GOOGLE_CLOUD_LOCATION=$Region"
) -join ","

gcloud run deploy $Service `
    --source . `
    --region $Region `
    --allow-unauthenticated `
    --port 8080 `
    --memory 1Gi `
    --cpu 1 `
    --timeout 300 `
    --min-instances 1 `
    --set-env-vars $plain

gcloud run services describe $Service --region $Region --format "value(status.url)"
