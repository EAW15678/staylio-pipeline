"""
tools/preview_agent5.py — Agent 5 local preview tool

Regenerates the landing-page HTML for an existing property using data
already cached in Redis/Supabase. Writes the result to a local file.

SAFE: Does NOT deploy, does NOT call paid services, does NOT provision
calendar sync, does NOT trigger any upstream agents.

Usage:
    python tools/preview_agent5.py <property_id>
    python tools/preview_agent5.py <property_id> --mock-curation path/to/curation.json

Options:
    --mock-curation <path>   Inject a local JSON file as image_curation into
                             visual_media before rendering. Use this to test
                             the LLM curation path without running the pipeline.

Examples:
    # Render using cached data only
    python tools/preview_agent5.py a1b2c3d4-0001-0001-0001-000000000001

    # Render with a mock curation result
    python tools/preview_agent5.py a1b2c3d4-0001-0001-0001-000000000001 \\
        --mock-curation tools/sample_curation.json

Output:
    /tmp/preview_<property_id>.html

Required env vars:
    REDIS_URL, REDIS_TOKEN
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   (only if Redis misses)
    ANTHROPIC_API_KEY                         (transitively imported — any value works)

Tip: load .env before running:
    env $(grep -v '^#' .env | xargs) python tools/preview_agent5.py <property_id>
"""

import json
import os
import sys

# ── Add repo root to path ──────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Parse arguments ────────────────────────────────────────────────────────────
def _parse_args() -> tuple[str, str | None]:
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: python tools/preview_agent5.py <property_id> [--mock-curation <path>]",
            file=sys.stderr,
        )
        sys.exit(1)

    property_id = args[0].strip()
    mock_curation_path: str | None = None

    i = 1
    while i < len(args):
        if args[i] == "--mock-curation" and i + 1 < len(args):
            mock_curation_path = args[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            sys.exit(1)

    return property_id, mock_curation_path

property_id, mock_curation_path = _parse_args()
print(f"\n[preview_agent5] property_id = {property_id}")
if mock_curation_path:
    print(f"[preview_agent5] mock_curation = {mock_curation_path}")

# ── Load mock curation JSON if provided ────────────────────────────────────────
mock_curation: dict | None = None
if mock_curation_path:
    try:
        with open(mock_curation_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Accept either the bare curation dict or a wrapped payload
        if "images" in raw and "property" in raw:
            mock_curation = {
                "status": "complete",
                "image_set_hash": "mock",
                "images": raw["images"],
                "property": raw["property"],
            }
        elif raw.get("status") == "complete" and "images" in raw:
            mock_curation = raw
        else:
            print(
                f"[preview_agent5] WARNING: {mock_curation_path} does not look like a valid "
                "curation JSON (needs 'images' and 'property' keys). Ignoring.",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[preview_agent5] ERROR: could not load mock curation: {exc}", file=sys.stderr)
        sys.exit(1)

# ── Imports (after path is set) ────────────────────────────────────────────────
from core.pipeline_status import get_cached_knowledge_base          # noqa: E402
from agents.agent5.page_builder import build_landing_page_html      # noqa: E402
from agents.agent5.agent import _load_from_cache_or_state, _fallback_slug  # noqa: E402

# ── Step 1: Load knowledge base ────────────────────────────────────────────────
print("\n[1/4] Loading knowledge base …")
kb = get_cached_knowledge_base(property_id)
if not kb:
    print(
        f"\n[preview_agent5] ERROR: No cached knowledge base found for "
        f"property_id={property_id!r}\n"
        "  Make sure the pipeline has run at least once for this property and\n"
        "  that REDIS_URL / REDIS_TOKEN are set correctly.\n",
        file=sys.stderr,
    )
    sys.exit(1)

kb_found = True
print(f"  KB found: YES  ({len(kb)} top-level keys)")

# ── Step 2: Load optional agent outputs ────────────────────────────────────────
print("\n[2/4] Loading cached agent outputs …")
_state: dict = {"property_id": property_id}

content_package = _load_from_cache_or_state(_state, property_id, "content_package", "agent2")
cp_found = bool(content_package)
print(f"  content_package  : {'YES' if cp_found else 'MISSING — copy sections will be empty'}")

visual_media = _load_from_cache_or_state(_state, property_id, "visual_media_package", "visual_media")
vm_found = bool(visual_media)
print(f"  visual_media     : {'YES' if vm_found else 'MISSING — gallery will be empty'}")

local_guide = _load_from_cache_or_state(_state, property_id, "local_guide", "local_guide")
lg_found = bool(local_guide)
print(f"  local_guide      : {'YES' if lg_found else 'MISSING — local section will be empty'}")

# ── Step 2b: Inject mock curation if provided ──────────────────────────────────
if mock_curation:
    if not visual_media:
        visual_media = {}
    visual_media["image_curation"] = mock_curation
    n_mock_images = len(mock_curation.get("images") or [])
    n_tour_sections = len((mock_curation.get("property") or {}).get("photo_tour_sections") or [])
    print(
        f"\n  [mock] image_curation injected: {n_mock_images} images, "
        f"{n_tour_sections} photo tour sections"
    )
    n_excluded   = sum(1 for r in mock_curation["images"] if r.get("role") == "exclude")
    n_dupes      = sum(1 for r in mock_curation["images"] if not r.get("is_primary_in_group", True))
    role_counts: dict[str, int] = {}
    for r in mock_curation["images"]:
        role = r.get("role") or "gallery_only"
        role_counts[role] = role_counts.get(role, 0) + 1
    print(f"  [mock] roles: {', '.join(f'{r}={n}' for r, n in sorted(role_counts.items()))}")
    print(f"  [mock] excluded={n_excluded}, non-primary(dupes)={n_dupes}")

# ── Step 3: Diagnostic counts ──────────────────────────────────────────────────
print("\n[3/4] Diagnostic counts …")

media_assets = visual_media.get("media_assets", []) if vm_found else []
n_assets = len(media_assets)
print(f"  Visual media assets     : {n_assets}")

# Curation path active?
image_curation = visual_media.get("image_curation") if visual_media else None
curation_active = (
    image_curation is not None
    and image_curation.get("status") == "complete"
    and bool(image_curation.get("images"))
)
print(f"  LLM curation active     : {'YES' + (' (mock)' if mock_curation else ' (cached)') if curation_active else 'NO — GCV fallback'}")

if curation_active:
    n_curation_images = len(image_curation.get("images") or [])
    n_tour = len((image_curation.get("property") or {}).get("photo_tour_sections") or [])
    print(f"  Curation images         : {n_curation_images}")
    print(f"  Photo tour sections     : {n_tour}")
else:
    # Estimate from media_assets using GCV path logic
    _CAT_TO_MODULE = {
        "exterior": "Exterior & Views", "view": "Exterior & Views",
        "pool_hot_tub": "Outdoor & Pool", "outdoor_entertaining": "Outdoor & Pool",
        "living_room": "Living Room", "game_entertainment": "Living Room",
        "kitchen": "Kitchen",
        "master_bedroom": "Bedrooms", "standard_bedroom": "Bedrooms",
        "bathroom": "Bathrooms",
        "local_area": "Amenities & Extras", "uncategorised": "Amenities & Extras",
    }
    modules_with_photos: set = set()
    for asset in media_assets:
        cat = asset.get("subject_category") or asset.get("category") or ""
        mod = _CAT_TO_MODULE.get(cat)
        if mod:
            modules_with_photos.add(mod)
    _OPTIONAL = {"Amenities & Extras"}
    _ALL_MODULES = [
        "Exterior & Views", "Outdoor & Pool", "Living Room",
        "Kitchen", "Bedrooms", "Bathrooms", "Amenities & Extras",
    ]
    estimated = len([m for m in _ALL_MODULES if m in modules_with_photos])
    print(f"  Photo tour modules (est): ~{estimated} / {len(_ALL_MODULES)}")

MAX_VISIBLE_ALL_PHOTOS = 30
visible_gallery = min(n_assets, MAX_VISIBLE_ALL_PHOTOS)
print(f"  All Photos visible      : {visible_gallery} (cap={MAX_VISIBLE_ALL_PHOTOS}, total={n_assets})")

# ── Step 4: Build slug and render HTML ────────────────────────────────────────
print("\n[4/4] Rendering HTML …")
slug     = kb.get("slug") or _fallback_slug(kb)
page_url = f"https://{slug}.staylio.ai"
print(f"  slug     = {slug!r}")
print(f"  page_url = {page_url}")

html = build_landing_page_html(
    kb=kb,
    content_package=content_package,
    visual_media=visual_media or {},
    local_guide=local_guide,
    page_url=page_url,
    slug=slug,
    calendar_cache_endpoint=None,
    api_base_url="https://staylio-pipeline-production.up.railway.app",
)

# ── Write output ───────────────────────────────────────────────────────────────
out_path = f"/tmp/preview_{property_id}.html"
with open(out_path, "w", encoding="utf-8") as fh:
    fh.write(html)

html_bytes = len(html.encode("utf-8"))

# ── Summary ────────────────────────────────────────────────────────────────────
curation_label = "YES (mock)" if mock_curation else ("YES (cached)" if curation_active else "NO")
print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preview complete
  ─────────────────────────────────────────────────
  Output file    : {out_path}
  HTML size      : {html_bytes:,} bytes ({html_bytes // 1024} KB)

  Data sources
  ─────────────────────────────────────────────────
  KB found               : {"YES" if kb_found else "NO"}
  content_package found  : {"YES" if cp_found else "NO (empty)"}
  visual_media found     : {"YES" if vm_found else "NO (empty)"}
  local_guide found      : {"YES" if lg_found else "NO (empty)"}
  LLM curation active    : {curation_label}

  Gallery
  ─────────────────────────────────────────────────
  Media assets           : {n_assets}
  All Photos visible     : {visible_gallery}

  SAFE — no deploy, no paid APIs, no upstream agents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open in browser:
  open {out_path}
""")
