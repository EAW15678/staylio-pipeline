"""
tools/deduped_review_sheet.py — Deduped image review sheet

Reads an existing image_review JSON, removes user-marked duplicate IDs,
and generates a clean HTML + JSON with the same review IDs and sheet order.

SAFE: reads local /tmp files only. No Redis, no Supabase, no API calls.

Usage:
    python tools/deduped_review_sheet.py <property_id> [--dupes ID,ID,ID,...]

If --dupes is omitted, the hardcoded DUPLICATE_IDS list below is used.

Output:
    /tmp/image_review_deduped_<property_id>.html
    /tmp/image_review_deduped_<property_id>.json
"""

import json
import os
import sys

# ── Duplicate IDs to remove (from Erick's review) ─────────────────────────────
DUPLICATE_IDS: set[str] = {
    "I02", "I05", "I08", "I10",
    "J01", "J03", "J07", "J09", "J12",
    "K04", "K05", "K07", "K08", "K09",
    "H02", "H04", "H05", "H08", "H09", "H10", "H12",
    "G04", "G05", "G06", "G08",
    "F02", "F06", "F12",
    "E03", "E05", "E06", "E08", "E11",
    "D04", "D05", "D09", "D12",
    "C04", "C06",
    "B03", "B06", "B07", "B08",
    "A02", "A06", "A07", "A08", "A10", "A12",
}

# ── Conflict flags — IDs marked as duplicates but also assigned roles ──────────
CONFLICTS: list[dict] = [
    {
        "id":     "K09",
        "reason": "Selected as Extras Supporting but also listed as Duplicate",
    },
    {
        "id":     "B03",
        "reason": "Selected as Exterior Ancillary but also listed as Duplicate",
    },
    {
        "id":     "H09",
        "reason": "Selected as Primary Bedroom Hero but also listed as Duplicate",
    },
    {
        "id":     "C08",
        "reason": "Appears as both Living Supporting and Living Ancillary (not a duplicate issue — role conflict)",
    },
]

# ── Args ───────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python tools/deduped_review_sheet.py <property_id> [--dupes ID,ID,...]",
          file=sys.stderr)
    sys.exit(1)

property_id = sys.argv[1].strip()

# Optional --dupes override
dupes_to_remove = DUPLICATE_IDS
i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--dupes" and i + 1 < len(sys.argv):
        dupes_to_remove = set(sys.argv[i + 1].split(","))
        i += 2
    else:
        i += 1

print(f"\n[deduped_review_sheet] property_id = {property_id}")
print(f"[deduped_review_sheet] duplicates to remove = {len(dupes_to_remove)}")

# ── Load source JSON ───────────────────────────────────────────────────────────
src_json = f"/tmp/image_review_{property_id}.json"
if not os.path.exists(src_json):
    print(
        f"\nERROR: {src_json} not found.\n"
        "  Run image_review_sheet.py first:\n"
        f"  python tools/image_review_sheet.py {property_id}",
        file=sys.stderr,
    )
    sys.exit(1)

with open(src_json, encoding="utf-8") as f:
    all_records: list[dict] = json.load(f)

print(f"[deduped_review_sheet] loaded {len(all_records)} records from {src_json}")

# ── Check conflicts ────────────────────────────────────────────────────────────
all_ids = {r["review_id"] for r in all_records}
conflict_ids   = {c["id"] for c in CONFLICTS}
missing_from_data = (dupes_to_remove | conflict_ids) - all_ids
if missing_from_data:
    print(f"  WARNING: these IDs not found in source data: {sorted(missing_from_data)}")

# ── Filter ────────────────────────────────────────────────────────────────────
kept   = [r for r in all_records if r["review_id"] not in dupes_to_remove]
removed = [r for r in all_records if r["review_id"] in dupes_to_remove]

print(f"  Original : {len(all_records)}")
print(f"  Removed  : {len(removed)}")
print(f"  Remaining: {len(kept)}")

# ── Write deduped JSON ────────────────────────────────────────────────────────
out_json = f"/tmp/image_review_deduped_{property_id}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2)
print(f"\n  JSON written: {out_json}")

# ── Build HTML ────────────────────────────────────────────────────────────────
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

def _card_html(r: dict, is_conflict: bool, conflict_note: str) -> str:
    colour      = _cat_colour(r["subject_category"])
    display_url = r.get("asset_url_enhanced") or r.get("asset_url_original") or ""
    labels      = ", ".join((r.get("labels_enhanced") or [])[:5]) or "—"
    filename    = (r.get("original_filename") or r.get("asset_url_original", "").split("/")[-1].split("?")[0])
    if len(filename) > 32:
        filename = "…" + filename[-29:]
    enhanced_badge = (
        '<span class="badge enhanced">enhanced</span>'
        if r.get("asset_url_enhanced")
        else '<span class="badge original">original only</span>'
    )
    conflict_banner = ""
    if is_conflict:
        conflict_banner = f'<div class="conflict-banner">⚠ {conflict_note}</div>'
    card_class = "card conflict-card" if is_conflict else "card"
    return f"""
    <div class="{card_class}">
      {conflict_banner}
      <div class="card-id">{r["review_id"]}</div>
      <a href="{display_url}" target="_blank" rel="noopener">
        <img src="{display_url}" alt="{r["review_id"]}" loading="lazy"
             onerror="this.style.background='#ddd';this.alt='load failed'">
      </a>
      <div class="meta">
        <div class="cat-badge" style="background:{colour};">{r["subject_category"]}</div>
        <div class="row"><span class="label">rank</span><span>{r.get("category_rank", "—")}</span></div>
        <div class="row"><span class="label">source</span><span>{r.get("source", "—")}</span></div>
        <div class="row"><span class="label">file</span><span class="filename">{filename}</span></div>
        <div class="row labels"><span class="label">labels</span><span>{labels}</span></div>
        <div class="badges-row">{enhanced_badge}</div>
      </div>
    </div>"""

# Group kept records by sheet letter, preserving original order
conflict_map = {c["id"]: c["reason"] for c in CONFLICTS}
sheets: dict[str, list] = {}
for r in kept:
    letter = r["review_id"][0]
    sheets.setdefault(letter, []).append(r)

sections_html = ""
for letter, sheet_records in sheets.items():
    start_id = sheet_records[0]["review_id"]
    end_id   = sheet_records[-1]["review_id"]
    cards = "".join(
        _card_html(r, r["review_id"] in conflict_map, conflict_map.get(r["review_id"], ""))
        for r in sheet_records
    )
    sections_html += f"""
  <section>
    <h2>Sheet {letter} &nbsp;<span class="sheet-range">({start_id}–{end_id}, {len(sheet_records)} remaining)</span></h2>
    <div class="grid">{cards}</div>
  </section>"""

# Conflict panel
conflict_items_html = "".join(
    f'<li><strong>{c["id"]}</strong> — {c["reason"]}</li>'
    for c in CONFLICTS
)

# Removed IDs panel (sorted)
removed_ids_sorted = sorted(r["review_id"] for r in removed)
removed_chips = "".join(
    f'<span class="chip">{rid}</span>'
    for rid in removed_ids_sorted
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Deduped Review — {property_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fa; color: #222;
      margin: 0; padding: 16px 24px 48px;
    }}
    header {{ border-bottom: 2px solid #ddd; padding-bottom: 12px; margin-bottom: 20px; }}
    header h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
    header p  {{ font-size: 0.82rem; color: #666; margin: 0; }}
    .panel {{
      background: #fff; border: 1px solid #dde0e8; border-radius: 8px;
      padding: 12px 16px; margin-bottom: 20px; font-size: 0.83rem; line-height: 1.7;
    }}
    .panel h3 {{ margin: 0 0 8px; font-size: 0.9rem; }}
    .conflict-panel {{ border-left: 4px solid #e07b39; }}
    .conflict-panel li {{ margin-bottom: 4px; }}
    .removed-panel {{ border-left: 4px solid #aaa; }}
    .chip {{
      display: inline-block; background: #eee; border-radius: 4px;
      padding: 1px 6px; margin: 2px; font-size: 0.75rem; font-family: monospace;
    }}
    h2 {{
      font-size: 1rem; font-weight: 600; margin: 28px 0 12px;
      padding: 6px 12px; background: #e8eaf0; border-radius: 6px;
    }}
    .sheet-range {{ font-size: 0.8rem; font-weight: 400; color: #666; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: #fff; border: 1px solid #dde0e8; border-radius: 8px;
      overflow: hidden; position: relative;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    .conflict-card {{ border: 2px solid #e07b39; }}
    .conflict-banner {{
      background: #fff3e0; color: #7c4a03;
      font-size: 0.7rem; padding: 4px 8px; line-height: 1.3;
      border-bottom: 1px solid #ffe0b2;
    }}
    .card-id {{
      position: absolute; top: 8px; left: 8px;
      background: rgba(0,0,0,.72); color: #fff;
      font-size: 0.78rem; font-weight: 700;
      padding: 2px 7px; border-radius: 4px; z-index: 2;
    }}
    .conflict-card .card-id {{ background: #c0392b; }}
    .card a {{ display: block; }}
    .card img {{
      width: 100%; height: 160px; object-fit: cover;
      display: block; background: #e0e0e0;
    }}
    .meta {{ padding: 8px 10px 10px; font-size: 0.75rem; }}
    .cat-badge {{
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      font-size: 0.72rem; font-weight: 600; margin-bottom: 6px;
    }}
    .row {{
      display: flex; gap: 4px; margin-bottom: 2px;
      line-height: 1.4; color: #444;
    }}
    .label {{ color: #999; flex-shrink: 0; width: 44px; }}
    .filename {{ word-break: break-all; color: #555; }}
    .labels {{ align-items: flex-start; }}
    .badges-row {{ margin-top: 6px; }}
    .badge {{ font-size: 0.68rem; padding: 1px 6px; border-radius: 3px; font-weight: 600; }}
    .badge.enhanced {{ background: #d4edda; color: #155724; }}
    .badge.original  {{ background: #fff3cd; color: #856404; }}
  </style>
</head>
<body>
<header>
  <h1>Deduped Image Review Sheet</h1>
  <p>property_id: <strong>{property_id}</strong>
     &nbsp;·&nbsp; {len(all_records)} original
     &nbsp;→&nbsp; <strong>{len(kept)} remaining</strong>
     &nbsp;·&nbsp; {len(removed)} removed
  </p>
</header>

<div class="panel conflict-panel">
  <h3>⚠ Conflicts ({len(CONFLICTS)} items require a decision)</h3>
  <ul>{conflict_items_html}</ul>
</div>

<div class="panel removed-panel">
  <h3>Removed IDs ({len(removed)})</h3>
  {removed_chips}
</div>

{sections_html}
</body>
</html>"""

out_html = f"/tmp/image_review_deduped_{property_id}.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML written: {out_html}")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Deduped review sheet complete
  ─────────────────────────────────────────────────
  Original count  : {len(all_records)}
  Duplicates removed: {len(removed)}  ({", ".join(removed_ids_sorted)})
  Remaining       : {len(kept)}

  Conflicts requiring decision
  ─────────────────────────────────────────────────""")
for c in CONFLICTS:
    print(f"  ⚠  {c['id']}: {c['reason']}")
print(f"""
  Output files
  ─────────────────────────────────────────────────
  HTML : {out_html}
  JSON : {out_json}

  SAFE — no writes to Redis/Supabase, no API calls
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open in browser:
  open {out_html}
""")
