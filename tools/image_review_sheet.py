"""
tools/image_review_sheet.py — Local image review sheet for property photos

Loads cached media_assets for a property and generates:
  1. /tmp/image_review_<property_id>.html  — visual grid for browser review
  2. /tmp/image_review_<property_id>.json  — ID mapping for use in sample_curation.json

Each image is assigned a simple review ID (A01, A02, ..., B01, B02, ...) that
matches the contact sheet labels used by the LLM curation system.

SAFE: reads Redis/Supabase cache only. No writes, no API calls, no pipeline.

Usage:
    python tools/image_review_sheet.py <property_id>

Example:
    env $(grep -v '^#' .env | xargs) \\
        python tools/image_review_sheet.py a1b2c3d4-0001-0001-0001-000000000001

Tip: copy the JSON output directly into tools/sample_curation.json to build
     a mock curation for use with tools/preview_agent5.py.
"""

import json
import os
import sys

# ── Repo root on path ──────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Args ───────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python tools/image_review_sheet.py <property_id>", file=sys.stderr)
    sys.exit(1)

property_id = sys.argv[1].strip()
print(f"\n[image_review_sheet] property_id = {property_id}")

# ── Imports ────────────────────────────────────────────────────────────────────
from agents.agent5.agent import _load_from_cache_or_state   # noqa: E402
from core.pipeline_status import get_cached_knowledge_base  # noqa: E402

# ── Load visual_media ─────────────────────────────────────────────────────────
print("[1/3] Loading cached visual_media …")
_state: dict = {"property_id": property_id}
visual_media = _load_from_cache_or_state(_state, property_id, "visual_media_package", "visual_media")

if not visual_media:
    print(
        "\nERROR: No cached visual_media found.\n"
        "  Make sure the pipeline has run for this property and\n"
        "  REDIS_URL / REDIS_TOKEN are set correctly.",
        file=sys.stderr,
    )
    sys.exit(1)

media_assets: list = visual_media.get("media_assets") or []
if not media_assets:
    print("ERROR: visual_media found but media_assets list is empty.", file=sys.stderr)
    sys.exit(1)

print(f"  Loaded {len(media_assets)} media assets")

# ── Assign review IDs ─────────────────────────────────────────────────────────
# Sort: category_rank within subject_category, then uncategorised last.
_CAT_ORDER = [
    "exterior", "view", "pool_hot_tub", "outdoor_entertaining",
    "living_room", "kitchen", "master_bedroom", "standard_bedroom",
    "bathroom", "game_entertainment", "local_area", "uncategorised",
]

def _sort_key(a: dict) -> tuple:
    cat = (a.get("subject_category") or "uncategorised").lower()
    try:
        pri = _CAT_ORDER.index(cat)
    except ValueError:
        pri = len(_CAT_ORDER)
    return (pri, a.get("category_rank") or 999)

sorted_assets = sorted(media_assets, key=_sort_key)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PER_SHEET = 12   # matches llm_curator.py contact sheet size

def _review_id(index: int) -> str:
    sheet  = index // _PER_SHEET
    cell   = index %  _PER_SHEET
    letter = _LETTERS[sheet % len(_LETTERS)]
    return f"{letter}{cell + 1:02d}"

# ── Build records ─────────────────────────────────────────────────────────────
records: list[dict] = []
for i, asset in enumerate(sorted_assets):
    rid          = _review_id(i)
    original_url = asset.get("asset_url_original") or ""
    enhanced_url = asset.get("asset_url_enhanced") or ""
    display_url  = enhanced_url or original_url
    original_filename = original_url.split("/")[-1].split("?")[0] if original_url else ""
    records.append({
        "review_id":          rid,
        "display_url":        display_url,
        "asset_url_original": original_url,
        "asset_url_enhanced": enhanced_url,
        "subject_category":   (asset.get("subject_category") or "uncategorised").lower(),
        "category_rank":      asset.get("category_rank") or 999,
        "source":             asset.get("source") or "unknown",
        "labels_enhanced":    asset.get("labels_enhanced") or [],
        "original_filename":  original_filename,
    })

# ── JSON output ────────────────────────────────────────────────────────────────
print("[2/3] Writing JSON mapping …")
json_path = f"/tmp/image_review_{property_id}.json"
json_records = [
    {
        "review_id":          r["review_id"],
        "asset_url_original": r["asset_url_original"],
        "asset_url_enhanced": r["asset_url_enhanced"],
        "subject_category":   r["subject_category"],
        "category_rank":      r["category_rank"],
        "source":             r["source"],
        "labels_enhanced":    r["labels_enhanced"],
    }
    for r in records
]
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_records, f, indent=2)
print(f"  Written: {json_path}")

# ── HTML output ────────────────────────────────────────────────────────────────
print("[3/3] Writing HTML review sheet …")
html_path = f"/tmp/image_review_{property_id}.html"

# Group by sheet letter for section headers
sheets: dict[str, list] = {}
for r in records:
    letter = r["review_id"][0]
    sheets.setdefault(letter, []).append(r)

# Category → pastel colour for badge
_CAT_COLOURS: dict[str, str] = {
    "exterior":             "#d4edda",
    "view":                 "#cce5ff",
    "pool_hot_tub":         "#b8d4f8",
    "outdoor_entertaining": "#d6eaf8",
    "living_room":          "#fff3cd",
    "kitchen":              "#fde8d8",
    "master_bedroom":       "#e8d5f5",
    "standard_bedroom":     "#f0e6ff",
    "bathroom":             "#d5f5e3",
    "game_entertainment":   "#fce4ec",
    "local_area":           "#f5f5dc",
    "uncategorised":        "#f0f0f0",
}

def _cat_colour(cat: str) -> str:
    return _CAT_COLOURS.get(cat, "#f0f0f0")

def _card_html(r: dict) -> str:
    colour    = _cat_colour(r["subject_category"])
    labels_str = ", ".join(r["labels_enhanced"][:5]) or "—"
    filename  = r["original_filename"] or "—"
    if len(filename) > 32:
        filename = "…" + filename[-29:]
    enhanced_badge = (
        '<span class="badge enhanced">enhanced</span>'
        if r["asset_url_enhanced"]
        else '<span class="badge original">original only</span>'
    )
    return f"""
    <div class="card">
      <div class="card-id">{r["review_id"]}</div>
      <a href="{r["display_url"]}" target="_blank" rel="noopener">
        <img src="{r["display_url"]}" alt="{r["review_id"]}" loading="lazy"
             onerror="this.style.background='#ddd';this.alt='failed to load'">
      </a>
      <div class="meta">
        <div class="cat-badge" style="background:{colour};">{r["subject_category"]}</div>
        <div class="row"><span class="label">rank</span><span>{r["category_rank"]}</span></div>
        <div class="row"><span class="label">source</span><span>{r["source"]}</span></div>
        <div class="row"><span class="label">file</span><span class="filename">{filename}</span></div>
        <div class="row labels"><span class="label">labels</span><span>{labels_str}</span></div>
        <div class="badges-row">{enhanced_badge}</div>
      </div>
    </div>"""

sections_html = ""
for letter, sheet_records in sheets.items():
    start_id = sheet_records[0]["review_id"]
    end_id   = sheet_records[-1]["review_id"]
    cards    = "".join(_card_html(r) for r in sheet_records)
    sections_html += f"""
  <section>
    <h2>Sheet {letter} &nbsp;<span class="sheet-range">({start_id} – {end_id}, {len(sheet_records)} images)</span></h2>
    <div class="grid">{cards}</div>
  </section>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Review — {property_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fa;
      color: #222;
      margin: 0;
      padding: 16px 24px 48px;
    }}
    header {{
      border-bottom: 2px solid #ddd;
      padding-bottom: 12px;
      margin-bottom: 24px;
    }}
    header h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
    header p  {{ font-size: 0.82rem; color: #666; margin: 0; }}
    h2 {{
      font-size: 1rem;
      font-weight: 600;
      margin: 28px 0 12px;
      padding: 6px 12px;
      background: #e8eaf0;
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .sheet-range {{ font-size: 0.8rem; font-weight: 400; color: #666; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #dde0e8;
      border-radius: 8px;
      overflow: hidden;
      position: relative;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
      transition: box-shadow .15s;
    }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,.12); }}
    .card-id {{
      position: absolute;
      top: 8px; left: 8px;
      background: rgba(0,0,0,.72);
      color: #fff;
      font-size: 0.78rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      letter-spacing: .04em;
      z-index: 2;
    }}
    .card a {{ display: block; }}
    .card img {{
      width: 100%;
      height: 160px;
      object-fit: cover;
      display: block;
      background: #e0e0e0;
    }}
    .meta {{
      padding: 8px 10px 10px;
      font-size: 0.75rem;
    }}
    .cat-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .row {{
      display: flex;
      gap: 4px;
      margin-bottom: 2px;
      line-height: 1.4;
      color: #444;
    }}
    .label {{
      color: #999;
      flex-shrink: 0;
      width: 44px;
    }}
    .filename {{ word-break: break-all; color: #555; }}
    .labels  {{ align-items: flex-start; }}
    .labels span:last-child {{ color: #555; }}
    .badges-row {{ margin-top: 6px; }}
    .badge {{
      font-size: 0.68rem;
      padding: 1px 6px;
      border-radius: 3px;
      font-weight: 600;
    }}
    .badge.enhanced {{ background: #d4edda; color: #155724; }}
    .badge.original  {{ background: #fff3cd; color: #856404; }}
    .summary {{
      background: #fff;
      border: 1px solid #dde0e8;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 24px;
      font-size: 0.83rem;
      line-height: 1.7;
    }}
    .summary strong {{ color: #333; }}
  </style>
</head>
<body>
<header>
  <h1>Image Review Sheet</h1>
  <p>property_id: <strong>{property_id}</strong> &nbsp;·&nbsp; {len(records)} images &nbsp;·&nbsp; {len(sheets)} sheets</p>
</header>

<div class="summary">
  <strong>How to use:</strong>
  Note the review IDs (A01, A02 …) of the images you want to include in a curation.
  Copy <code>/tmp/image_review_{property_id}.json</code> to build a
  <code>sample_curation.json</code>, then run:<br>
  <code>python tools/preview_agent5.py {property_id} --mock-curation tools/sample_curation.json</code>
</div>

{sections_html}
</body>
</html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  Written: {html_path}")

# ── Summary ────────────────────────────────────────────────────────────────────
n_enhanced = sum(1 for r in records if r["asset_url_enhanced"])
cat_counts: dict[str, int] = {}
for r in records:
    cat_counts[r["subject_category"]] = cat_counts.get(r["subject_category"], 0) + 1
cat_summary = ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items()))

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Image review sheet complete
  ─────────────────────────────────────────────────
  Total images    : {len(records)}
  Enhanced        : {n_enhanced} / {len(records)}
  Sheets          : {len(sheets)} (Sheet {"–".join(list(sheets.keys())[[0,-1]])})
  Categories      : {cat_summary}

  Output files
  ─────────────────────────────────────────────────
  HTML  : {html_path}
  JSON  : {json_path}

  SAFE — no writes, no paid APIs, no pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open in browser:
  open {html_path}
""")
