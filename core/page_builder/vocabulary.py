"""
Section vocabulary for the substrate-native page builder.

Created 2026-08-19 (PAGE-2). Ruled by Erick.

The old GCV vocabulary (_CURATION_VALID_CATEGORIES in agents/agent5/page_builder.py)
used 12 internal category codes (exterior, view, pool_hot_tub, outdoor_entertaining,
living_room, kitchen, master_bedroom, standard_bedroom, bathroom, game_entertainment,
local_area, uncategorised) with a silent catch-all that dumped anything unrecognised
into "uncategorised." The display-name table mapped these codes to labels like
"Exterior & Views", "Outdoor & Pool", "Amenities & Extras" — names that did not
match the real section names already written by the curator.

This module replaces all of that. The seven section names below are the source of
truth, matching observations.curated_section exactly. Functions receive
curated_section directly — no mapping through subject_category or GCV names.

An unrecognised value does not silently become a catch-all bucket. It routes to
Extras and is logged — consistent with "downgrade never drop" (GUARDRAIL-2).

Default ordering: Exterior, Pool, Living Areas, Kitchen, Bedrooms, Bathrooms,
Extras. This is a default, not a ruling — Erick can correct it.
"""

import logging

logger = logging.getLogger(__name__)

# ── The seven sections ───────────────────────────────────────────────────────

SECTIONS: list[str] = [
    "Exterior",
    "Pool",
    "Living Areas",
    "Kitchen",
    "Bedrooms",
    "Bathrooms",
    "Extras",
]

SECTION_SET: frozenset = frozenset(SECTIONS)

# Section name IS the display header. No translation table.
SECTION_ORDER: dict[str, int] = {name: i for i, name in enumerate(SECTIONS)}

# ── Near-duplicate suppression: strict categories ────────────────────────────
# Bathrooms and Bedrooms are highly repetitive room types; use a stricter
# per-cluster cap (1 instead of 2).
# Old GCV: frozenset({"bathroom", "master_bedroom", "standard_bedroom"})
# master_bedroom and standard_bedroom collapsed into Bedrooms.

STRICT_DUPE_SECTIONS: frozenset = frozenset({"Bathrooms", "Bedrooms"})

# ── Module scoring: preferred and penalty labels ─────────────────────────────
# These are Google Vision API labels (like "sofa", "pool", "bed"), not section
# names. Re-keyed from old display names to real section names.

MODULE_PREFERRED_LABELS: dict[str, frozenset] = {
    "Exterior":     frozenset({"exterior", "house", "facade", "balcony", "porch",
                               "view", "ocean", "horizon", "sunset", "skyline"}),
    "Pool":         frozenset({"pool", "swimming pool", "deck", "hot tub", "spa",
                               "seating", "lounge chair", "patio", "pergola"}),
    "Living Areas": frozenset({"living room", "sofa", "couch", "great room",
                               "family room", "fireplace", "sitting area", "sectional"}),
    "Kitchen":      frozenset({"kitchen", "island", "countertop", "refrigerator",
                               "stove", "oven", "dining table", "dining room", "sink",
                               "cabinet"}),
    "Bedrooms":     frozenset({"bedroom", "bed", "king", "bunk", "primary bedroom",
                               "master bedroom", "pillow", "nightstand", "headboard"}),
    "Bathrooms":    frozenset({"bathroom", "vanity", "bathtub", "soaking tub",
                               "marble", "mirror", "double sink", "freestanding tub"}),
    "Extras":       frozenset({"game room", "gym", "office", "library",
                               "wine cellar", "bar", "garage", "laundry room"}),
}

MODULE_PENALTY_LABELS: dict[str, frozenset] = {
    "Kitchen":   frozenset({"laundry", "washer", "dryer", "closet", "hallway", "corridor",
                            "washing machine", "coffee maker", "coffee cup"}),
    "Bathrooms": frozenset({"shower head", "showerhead", "shower curtain",
                            "soap dispenser", "towel rack"}),
}

CLOSEUP_LABELS: frozenset = frozenset({
    "coffee maker", "coffee cup", "throw pillow", "candle", "vase",
    "plant", "artwork", "picture frame", "book", "remote control",
    "lamp shade", "decorative bowl", "fruit bowl",
})

DEPRIORITIZED_LABELS: frozenset = frozenset({
    "laundry", "washing machine", "dryer", "garage", "garage door",
    "parking lot", "hallway", "corridor", "closet", "storage room",
})

# Modules whose absence is silently skipped (no log warning).
OPTIONAL_SECTIONS: frozenset = frozenset({"Extras"})

# ── Per-section cap ──────────────────────────────────────────────────────────
MAX_IMAGES_PER_SECTION = 8

# ── Near-duplicate suppression constants ─────────────────────────────────────
NEAR_DUPE_LABEL_THRESHOLD = 0.4
MAX_PER_DUPE_CLUSTER = 2
SIMILARITY_PENALTY_WEIGHT = 0.3
MODULE_MAX_SIMILARITY = 0.35

# Photo Tour roles
PHOTO_TOUR_ROLES: frozenset = frozenset({"hero", "supporting"})

# Max visible supporting in Photo Tour module grid
MAX_VISIBLE_SUPPORTING = 3


def normalise_section(raw: str) -> str:
    """Map a curated_section value to one of the seven canonical sections.

    Known values pass through unchanged. Unrecognised values route to Extras
    and are logged — consistent with "downgrade never drop" (GUARDRAIL-2).
    """
    if raw in SECTION_SET:
        return raw
    logger.warning(
        "[page_builder] Unrecognised curated_section %r — routing to Extras "
        "(downgrade never drop, GUARDRAIL-2)",
        raw,
    )
    return "Extras"
