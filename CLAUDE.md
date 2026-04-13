# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Start the FastAPI server locally
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Trigger a pipeline run (requires running server)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"property_id": "<uuid>"}'

# Poll pipeline status
curl http://localhost:8000/status/<property_id>
```

## Tests

```bash
# Run all tests
pytest

# Run tests for a specific agent
pytest agents/agent1/tests/
pytest agents/agent2/tests/ -v

# Run a single test file
pytest agents/agent3/tests/test_claid.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

Node.js is also required for two subprocess workers: `agents/agent3/crop_worker.js` (photo crops) and `agents/agent7/puppeteer_renderer.js` (PDF generation).

---

## Architecture

### Pipeline Overview

A 7-agent LangGraph pipeline triggered via POST `/run` or `/intake`. Agents execute in this order:

```
Agent 1 (Content Ingestion)
  ↓
Agent 2 (Content Enhancement) ─┐
Agent 3 (Visual Media)         ├─ parallel
Agent 4 (Local Discovery)     ─┘
  ↓
Agent 5 (Website Builder)      ← waits for 2, 3, 4
  ↓
Agent 6 (Social Media)  ─┐
Agent 7 (Analytics)     ─┘  parallel
```

The graph is defined in `pipeline/graph.py` as a LangGraph `StateGraph`. State is a `PipelineState` TypedDict with reducer functions: scalar fields use `_last()` (last-write-wins) and list fields use `_concat()` (accumulate) to avoid `InvalidUpdateError` during parallel merges.

### Inter-Agent Communication

Agents share data two ways:
1. **Redis cache** — written by producing agent, read by consuming agent. Key pattern: `{property_id}:{label}` (e.g. `{property_id}:agent2`, `{property_id}:visual_media`, `{property_id}:local_guide`).
2. **LangGraph state dict** — fallback if Redis cache miss.

`core/pipeline_status.py` provides `cache_knowledge_base()` / `get_cached_knowledge_base()` for all cache reads/writes, and `update_pipeline_status()` which dual-writes to Redis + Supabase for dashboard polling.

### Central Data Model

`models/property.py` defines `PropertyKnowledgeBase` — the canonical property record built by Agent 1. Each field is a `PropertyField` wrapper with `value`, `source` (enum), and `confidence` (0–1). Merge policy: intake portal data always wins; among scraped sources, higher confidence wins.

### Agent Responsibilities

| Agent | Key Output | Stores To |
|-------|-----------|-----------|
| 1 | `PropertyKnowledgeBase` | Supabase `property_knowledge_bases` + Redis |
| 2 | `ContentPackage` (copy, FAQs, social captions) | Supabase `content_versions` + Redis `:agent2` |
| 3 | `VisualMediaPackage` (R2 URLs, asset metadata) | R2 buckets + Supabase `media_assets` + Redis `:visual_media` |
| 4 | `LocalGuide` (venues by category) | Redis `:local_guide` |
| 5 | `LandingPage` (deployed HTML) | Supabase `landing_pages`, page URL cached for Agent 6 |
| 6 | `ContentCalendar`, `MetaCampaign` | Supabase `social_posts`, `meta_campaigns` |
| 7 | `AnalyticsSnapshot`, pixel snippet | Supabase `analytics_snapshots`, R2 (PDFs) |

### Photo Pipeline (Agent 3)

Photos flow: download → SHA-256 dedup → PIL resize if >9MB → R2 upload (original) → Claid.ai enhancement → R2 upload (enhanced) → Vision API tagging → category winner selection → social crops.

Claid.ai has a hard governance whitelist enforced in `validate_operations()` before every API call. Prohibited operations (background removal, generative fill, virtual staging, etc.) raise `ValueError` and are never submitted. This is non-negotiable — California AB 723 / FTC / NAR compliance.

### Deployment (Cloudflare)

Agent 5 deploys via `agents/agent5/cloudflare_deployer.py` — uploads `{slug}/index.html` to Cloudflare R2 (`R2_BUCKET_NAME`). The `staylio-router` Cloudflare Worker serves `*.staylio.ai` subdomains from R2. Cloudflare Pages is **not** used (paid feature).

### LLM Usage

- **Claude Sonnet 4.6** — landing page copy generation, quality review, KB normalization
- **Claude Haiku 4.5** — social captions, monthly report narratives
- **GPT-4o** — fallback if Sonnet is rate-limited (Agent 2)

Model IDs: `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`

### Key Environment Variables

```
SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
REDIS_URL, REDIS_TOKEN
ANTHROPIC_API_KEY, OPENAI_API_KEY
APIFY_API_TOKEN, APIFY_AIRBNB_ACTOR_ID, APIFY_VRBO_ACTOR_ID
FIRECRAWL_API_KEY
GOOGLE_PLACES_API_KEY
GOOGLE_SERVICE_ACCOUNT_JSON      # inline JSON for Vision API credentials
CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_BASE
CLAID_API_KEY
AYRSHARE_API_KEY
META_ACCESS_TOKEN, META_BUSINESS_ID
YELP_FUSION_API_KEY
```

Railway injects `PORT` at runtime. The app binds to `0.0.0.0:$PORT`.

### Scraper Notes

- **Apify** (Agent 1): Airbnb actor `tri_angle~airbnb-rooms-urls-scraper`. Actor input is passed directly as `json=actor_input` (not wrapped in `{"input": ...}`). Airbnb listings return photos under `images[].imageUrl`, not `photos[].url`.
- **Firecrawl** (Agent 1): Uses `/scrape` endpoint (free tier). Returns markdown parsed by `claude_parser.py`. The `/extract` endpoint is paid-only.
- **Google Places** (Agent 4): Uses Places API (New) — POST `/places:searchNearby`, GET `/places/{id}`. Auth via `X-Goog-Api-Key` + `X-Goog-FieldMask` headers. The legacy Places API endpoints are not used.
