"""
Gallery scoring, near-duplicate suppression, and section grouping.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/page_builder.py, vocabulary-corrected to use the real
seven-section taxonomy via curated_section instead of subject_category.

No database access. All functions are purely computational.
"""

import logging
import os
import re

from core.page_builder.vocabulary import (
    SECTIONS, SECTION_ORDER, SECTION_SET,
    STRICT_DUPE_SECTIONS,
    MODULE_PREFERRED_LABELS, MODULE_PENALTY_LABELS,
    CLOSEUP_LABELS, DEPRIORITIZED_LABELS,
    OPTIONAL_SECTIONS,
    MAX_IMAGES_PER_SECTION,
    NEAR_DUPE_LABEL_THRESHOLD, MAX_PER_DUPE_CLUSTER,
    SIMILARITY_PENALTY_WEIGHT, MODULE_MAX_SIMILARITY,
    MAX_VISIBLE_SUPPORTING,
    normalise_section,
)
from core.page_builder.helpers import _esc

logger = logging.getLogger(__name__)


# ── Similarity functions ─────────────────────────────────────────────────────

def _jaccard(a, b):
    """Jaccard similarity of two frozensets (0.0–1.0).
    Returns 0.0 when either set is empty (incomparable, not identical).
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _filename_seq_num(url):
    """Extract the sequential upload number from an R2 photo filename.
    Filename pattern: photo_NNN_<hash>.jpg  → returns NNN as int.
    Returns None when the pattern is not found.
    """
    try:
        stem = os.path.splitext(os.path.basename(url))[0]
        m = re.match(r"^photo_(\d+)", stem)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _caption_word_overlap(labels_a, labels_b):
    """Word-level Jaccard similarity between two label lists.
    Splits each label string into individual words (>2 chars) to catch
    partial matches that label-level Jaccard misses.
    Returns 0.0 when either tokenised set is empty.
    """
    def _tokenize(labels):
        words = set()
        for label in labels:
            for word in label.lower().split():
                if len(word) > 2:
                    words.add(word)
        return words

    ta = _tokenize(labels_a)
    tb = _tokenize(labels_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _combined_similarity(label_set_a, label_set_b, asset_a, asset_b):
    """Multi-signal similarity score between two assets (0.0–1.0).

    Weights:
      0.60  Vision label Jaccard (frozenset)
      0.25  Word-level caption overlap
      0.15  Filename sequential proximity (same source, ≤3 photos apart)
    """
    label_sim = _jaccard(label_set_a, label_set_b)
    word_sim = _caption_word_overlap(
        asset_a.get("labels_enhanced") or asset_a.get("labels_original") or [],
        asset_b.get("labels_enhanced") or asset_b.get("labels_original") or [],
    )
    fname_sim = 0.0
    src_a = (asset_a.get("source") or "").lower()
    src_b = (asset_b.get("source") or "").lower()
    if src_a and src_a == src_b:
        url_a = asset_a.get("url") or asset_a.get("asset_url_enhanced") or ""
        url_b = asset_b.get("url") or asset_b.get("asset_url_enhanced") or ""
        n_a = _filename_seq_num(url_a)
        n_b = _filename_seq_num(url_b)
        if n_a is not None and n_b is not None and abs(n_a - n_b) <= 3:
            fname_sim = 1.0
    return min(1.0, label_sim * 0.60 + word_sim * 0.25 + fname_sim * 0.15)


# ── Scoring ──────────────────────────────────────────────────────────────────

def _asset_score(asset):
    """Quality score for a single asset dict (higher = better).

    Weights:
      0.50  category_rank (inverted)
      0.30  composition_score from Vision API
      0.15  enhanced URL available
      0.05  source priority
      ×0.7  penalty for deprioritized labels
    """
    rank = asset.get("category_rank") or 999
    rank_score = 1.0 / rank
    comp = float(asset.get("composition_score") or 0.0)
    has_enhanced = 1.0 if asset.get("has_enhanced") else 0.0
    _SRC = {
        "intake_upload": 0.3,
        "pmc_website": 0.2,
        "vrbo_scraped": 0.1,
        "airbnb_scraped": 0.0,
        "unknown": 0.0,
    }
    src = _SRC.get((asset.get("source") or "unknown").lower(), 0.0)
    score = rank_score * 0.5 + comp * 0.3 + has_enhanced * 0.15 + src * 0.05

    labels = set(lbl.lower() for lbl in (asset.get("labels_enhanced") or []))
    if labels & DEPRIORITIZED_LABELS:
        score *= 0.7

    return score


def _module_quality_score(asset: dict, section_name: str) -> float:
    """Section-level quality score layered on top of _asset_score.

    Multipliers:
      ×1.25  preferred labels present
      ×0.40  penalty labels present (wrong-category content)
      ×0.80  tight decorative closeup labels
    """
    base = _asset_score(asset)
    labels = set(lbl.lower() for lbl in (asset.get("labels_enhanced") or []))

    preferred = MODULE_PREFERRED_LABELS.get(section_name, frozenset())
    penalty_set = MODULE_PENALTY_LABELS.get(section_name, frozenset())

    if labels & preferred:
        base *= 1.25
    if labels & penalty_set:
        base *= 0.4
    if labels & CLOSEUP_LABELS:
        base *= 0.8

    return base


# ── Near-duplicate suppression ───────────────────────────────────────────────
# This function logs whenever it actually removes something, so real run data
# can answer whether it's still doing useful work now that deduplicate.py
# handles duplicates upstream.

def _suppress_near_dupes(assets):
    """Within a single section, cluster images by multi-signal similarity
    and select diverse representatives.

    Per-section cap:
      STRICT_DUPE_SECTIONS (Bathrooms/Bedrooms): 1 per cluster
      all others: MAX_PER_DUPE_CLUSTER (default 2)

    Returns (kept_assets, n_dupes_removed, n_clusters).
    """
    if not assets:
        return assets, 0, 0

    section = normalise_section(assets[0].get("section") or "Extras")
    max_per_cluster = 1 if section in STRICT_DUPE_SECTIONS else MAX_PER_DUPE_CLUSTER

    # Phase 1: cluster assignment
    sorted_assets = sorted(assets, key=_asset_score, reverse=True)
    clusters = []
    founder_label_sets = []
    founder_assets = []

    for asset in sorted_assets:
        raw_labels = asset.get("labels_enhanced") or asset.get("labels_original") or []
        label_set = frozenset(lbl.lower() for lbl in raw_labels)

        matched_cluster = None
        best_sim = 0.0
        for i, founder_set in enumerate(founder_label_sets):
            sim = _combined_similarity(label_set, founder_set, asset, founder_assets[i])
            if sim >= NEAR_DUPE_LABEL_THRESHOLD and sim > best_sim:
                best_sim = sim
                matched_cluster = i

        if matched_cluster is not None:
            clusters[matched_cluster].append(asset)
        else:
            clusters.append([asset])
            founder_label_sets.append(label_set)
            founder_assets.append(asset)

    # Phase 2: diversity-aware selection
    kept = []
    n_dupes = 0

    for cluster in clusters:
        if len(cluster) == 1:
            kept.extend(cluster)
            continue

        label_sets = []
        raw_scores = []
        for a in cluster:
            raw_lbl = a.get("labels_enhanced") or a.get("labels_original") or []
            label_sets.append(frozenset(lbl.lower() for lbl in raw_lbl))
            raw_scores.append(_asset_score(a))

        selected_indices = []
        remaining = list(range(len(cluster)))

        for _ in range(max_per_cluster):
            if not remaining:
                break
            best_idx = None
            best_adj = -1.0
            for ri in remaining:
                if not selected_indices:
                    adj = raw_scores[ri]
                else:
                    max_sim = max(
                        _combined_similarity(
                            label_sets[ri], label_sets[si], cluster[ri], cluster[si]
                        )
                        for si in selected_indices
                    )
                    adj = raw_scores[ri] - max_sim * SIMILARITY_PENALTY_WEIGHT
                if adj > best_adj:
                    best_adj = adj
                    best_idx = ri
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        for i, a in enumerate(cluster):
            if i in selected_indices:
                kept.append(a)
            else:
                n_dupes += 1

    n_clusters = len(clusters)

    if n_dupes > 0:
        logger.info(
            "[page_builder] _suppress_near_dupes: section=%s removed=%d clusters=%d kept=%d. "
            "Logging to determine if this function still does useful work now that "
            "deduplicate.py handles duplicates upstream.",
            section, n_dupes, n_clusters, len(kept),
        )

    return kept, n_dupes, n_clusters


# ── Gallery item preparation (GCV heuristic fallback) ────────────────────────

def _prepare_gallery_items(
    media_assets: list,
    hero_photo: str,
    kb_photos: list,
    property_name: str,
) -> list:
    """Build gallery items from media_assets using curated_section.

    GCV heuristic fallback path — moved as-is, vocabulary-corrected.
    No overall cap; per-section cap of MAX_IMAGES_PER_SECTION (8).
    """
    if not media_assets and not kb_photos:
        return []

    raw_assets = []
    total_candidates = 0
    total_dupes_removed = 0
    total_clusters = 0
    cat_cluster_info = {}

    if media_assets:
        for rank, asset in enumerate(media_assets, 1):
            url = asset.get("asset_url_enhanced") or asset.get("asset_url_original") or ""
            if not url or url == hero_photo:
                continue

            section = normalise_section(asset.get("curated_section") or "Extras")

            labels = asset.get("labels_enhanced") or []
            alt = (
                f"{property_name} \u2013 " + ", ".join(labels[:3])
                if labels else f"{property_name} photo"
            )
            raw_assets.append({
                "url": url,
                "alt": alt,
                "category": section,
                "section": section,
                "rank": rank,
                "labels_enhanced": labels,
                "composition_score": asset.get("composition_score") or 0.0,
                "has_enhanced": bool(asset.get("asset_url_enhanced")),
                "source": asset.get("source") or "unknown",
            })

        total_candidates = len(raw_assets)

        # Near-dupe suppression per section
        by_section = {}
        for item in raw_assets:
            by_section.setdefault(item["section"], []).append(item)

        suppressed = []
        for sec, items in by_section.items():
            kept, n_removed, n_cl = _suppress_near_dupes(items)
            suppressed.extend(kept)
            total_dupes_removed += n_removed
            total_clusters += n_cl
            cat_cluster_info[sec] = (n_cl, n_removed)

        raw_assets = suppressed

    elif kb_photos:
        for rank, photo in enumerate(kb_photos, 1):
            url = ""
            if isinstance(photo, dict):
                url = photo.get("url") or ""
            elif isinstance(photo, str):
                url = photo
            if not url or url == hero_photo:
                continue
            raw_assets.append({
                "url": url,
                "alt": f"{property_name} photo",
                "category": "Extras",
                "section": "Extras",
                "rank": rank,
                "labels_enhanced": [],
                "composition_score": 0.0,
                "has_enhanced": False,
                "source": "unknown",
            })
        total_candidates = len(raw_assets)

    # Sort by (section_priority, rank)
    def _sort_key(item):
        sec = item["section"]
        priority = SECTION_ORDER.get(sec, len(SECTIONS))
        return (priority, item["rank"])

    raw_assets.sort(key=_sort_key)

    # Per-section balancing — no overall cap
    selected = []
    per_sec_count = {}
    remainder = []
    for item in raw_assets:
        sec = item["section"]
        if per_sec_count.get(sec, 0) < MAX_IMAGES_PER_SECTION:
            selected.append(item)
            per_sec_count[sec] = per_sec_count.get(sec, 0) + 1
        else:
            remainder.append(item)

    # Second pass: remaining items go in (no overall cap)
    selected.extend(remainder)

    # Logging
    sec_counts = {}
    for item in selected:
        sec_counts[item["section"]] = sec_counts.get(item["section"], 0) + 1
    sorted_secs = sorted(
        sec_counts.items(),
        key=lambda kv: SECTION_ORDER.get(kv[0], len(SECTIONS)),
    )
    sec_summary = ", ".join(f"{s}={n}" for s, n in sorted_secs)
    logger.info(
        "[page_builder] Gallery selection: candidates=%d, "
        "near-dupes suppressed=%d, clusters=%d, after-dedup=%d, final=%d. "
        "Sections: %s",
        total_candidates,
        total_dupes_removed,
        total_clusters,
        total_candidates - total_dupes_removed,
        len(selected),
        sec_summary,
    )

    return [
        {
            "url": i["url"],
            "alt": i["alt"],
            "category": i["section"],
            "section": i["section"],
            "rank": i["rank"],
            "labels_enhanced": i.get("labels_enhanced") or [],
            "composition_score": i.get("composition_score") or 0.0,
            "has_enhanced": i.get("has_enhanced", False),
            "source": i.get("source") or "unknown",
        }
        for i in selected
    ]


# ── Photo Tour module building ───────────────────────────────────────────────

def _build_category_modules(gallery_items: list) -> dict:
    """Build curated section modules from the flat gallery items list.

    Groups items by section and selects 1 hero + 2 supporting per section.
    Returns ordered dict: {section_name: {hero, supporting, all}}.
    """
    if not gallery_items:
        return {}

    # Group by section
    by_section = {}
    for item in gallery_items:
        sec = item.get("section") or item.get("category") or "Extras"
        by_section.setdefault(sec, []).append(item)

    result = {}
    skipped = []

    for section_name in SECTIONS:
        section_items = by_section.get(section_name, [])
        if not section_items:
            if section_name not in OPTIONAL_SECTIONS:
                skipped.append(section_name)
            continue

        # Score candidates
        scored = sorted(
            section_items,
            key=lambda x: _module_quality_score(x, section_name),
            reverse=True,
        )

        hero = None
        supporting = []
        selected_items = []

        for candidate in scored:
            if hero is None:
                hero = candidate
                selected_items.append(candidate)
                continue

            if len(supporting) >= 2:
                break

            # Check similarity against all selected
            cand_labels = frozenset(
                lbl.lower() for lbl in (candidate.get("labels_enhanced") or [])
            )
            too_similar = False
            for sel in selected_items:
                sel_labels = frozenset(
                    lbl.lower() for lbl in (sel.get("labels_enhanced") or [])
                )
                if _combined_similarity(cand_labels, sel_labels, candidate, sel) > MODULE_MAX_SIMILARITY:
                    too_similar = True
                    break

            if not too_similar:
                supporting.append(candidate)
                selected_items.append(candidate)

        if hero:
            result[section_name] = {
                "hero": hero,
                "supporting": supporting,
                "all": section_items,
            }

    logger.info(
        "[page_builder] Photo tour: %d modules built (%s); skipped: %s",
        len(result),
        ", ".join(
            f"{label}={len(mod['all'])}img"
            for label, mod in result.items()
        ),
        ", ".join(skipped) if skipped else "none",
    )

    return result
