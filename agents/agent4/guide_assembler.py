"""
Local Guide Assembler

Merges Google Places and Yelp Fusion results, deduplicates,
applies vibe-profile filtering to surface primary recommendations,
layers in owner Don't Miss picks, and generates the area introduction.

Merge policy (TS-11):
  - Business appears in both sources: create one Place record using
    Google for name/address/hours, Yelp for price tier and review count
  - Business in Google only: use Google data
  - Business in Yelp only: use Yelp data

Vibe filter categories define which PlaceCategories surface as
primary recommendations for each of the 6 vibe archetypes.
All categories are available in the full category browser regardless.
"""

import logging
import os
import re
from difflib import SequenceMatcher
from typing import Optional

import anthropic

from agents.agent4.models import (
    DataSource,
    DontMissPick,
    LocalGuide,
    Place,
    PlaceCategory,
)
from models.property import VibeProfile

logger = logging.getLogger(__name__)

# ── Vibe → Primary Category Priorities ───────────────────────────────────
# Defines which categories surface as PRIMARY recommendations (visual cards)
# and their order. All categories still appear in the full browser.

VIBE_PRIMARY_CATEGORIES: dict[str, list[PlaceCategory]] = {
    VibeProfile.ROMANTIC_ESCAPE: [
        PlaceCategory.EAT_AND_DRINK,      # Intimate restaurants
        PlaceCategory.COFFEE_CAFES,       # Morning rituals
        PlaceCategory.WELLNESS,           # Spa day
        PlaceCategory.ATTRACTIONS,        # Scenic spots
        PlaceCategory.ARTS_CULTURE,       # Evening culture
        PlaceCategory.NIGHTLIFE,          # Quiet bars
    ],
    VibeProfile.FAMILY_ADVENTURE: [
        PlaceCategory.FAMILY_KIDS,        # Kid-specific activities
        PlaceCategory.ADVENTURE_OUTDOORS, # Family hikes, parks
        PlaceCategory.EAT_AND_DRINK,      # Family restaurants
        PlaceCategory.ATTRACTIONS,        # Things to see
        PlaceCategory.TOURS_EXPERIENCES,  # Guided activities
        PlaceCategory.COFFEE_CAFES,       # Morning fuel
    ],
    VibeProfile.MULTIGENERATIONAL: [
        PlaceCategory.EAT_AND_DRINK,      # Group dinners
        PlaceCategory.ATTRACTIONS,        # Works for all ages
        PlaceCategory.FAMILY_KIDS,        # Grandkids
        PlaceCategory.ADVENTURE_OUTDOORS, # Easy walks, parks
        PlaceCategory.ARTS_CULTURE,       # Multigenerational interest
        PlaceCategory.SHOPPING,           # Rainy day option
    ],
    VibeProfile.WELLNESS_RETREAT: [
        PlaceCategory.WELLNESS,           # Spas, yoga
        PlaceCategory.ADVENTURE_OUTDOORS, # Hiking, nature
        PlaceCategory.COFFEE_CAFES,       # Quiet mornings
        PlaceCategory.EAT_AND_DRINK,      # Healthy dining
        PlaceCategory.ATTRACTIONS,        # Scenic overlooks
        PlaceCategory.ARTS_CULTURE,       # Contemplative spaces
    ],
    VibeProfile.ADVENTURE_BASE_CAMP: [
        PlaceCategory.ADVENTURE_OUTDOORS, # Primary reason they're here
        PlaceCategory.TOURS_EXPERIENCES,  # Guided adventures
        PlaceCategory.EAT_AND_DRINK,      # Fueling up
        PlaceCategory.COFFEE_CAFES,       # Early starts
        PlaceCategory.SHOPPING,           # Gear and supplies
        PlaceCategory.ATTRACTIONS,        # Scenic stops
    ],
    VibeProfile.SOCIAL_CELEBRATIONS: [
        PlaceCategory.NIGHTLIFE,          # Bars, clubs, live music
        PlaceCategory.EAT_AND_DRINK,      # Group dinners
        PlaceCategory.TOURS_EXPERIENCES,  # Group activities
        PlaceCategory.ATTRACTIONS,        # Photo spots
        PlaceCategory.ARTS_CULTURE,       # Live shows
        PlaceCategory.COFFEE_CAFES,       # Recovery mornings
    ],
}

# Number of primary recommendations to surface per property
PRIMARY_RECOMMENDATIONS_COUNT = 10

# Deduplication: names are considered the same if they have
# Jaro-Winkler similarity above this threshold
DEDUP_SIMILARITY_THRESHOLD = 0.88


def assemble_local_guide(
    property_id: str,
    google_places: list[Place],
    yelp_places: list[Place],
    vibe_profile: str,
    owner_dont_miss: list[str],
    neighborhood_description: Optional[str],
    city: Optional[str],
    state: Optional[str],
    anthropic_client: Optional[anthropic.Anthropic] = None,
) -> LocalGuide:
    """
    Main assembly function.

    1. Merge and deduplicate Google + Yelp results
    2. Apply vibe filter to surface primary recommendations
    3. Layer in owner Don't Miss picks
    4. Organise all places into category browser
    5. Generate area introduction via Claude
    """
    location_name = f"{city or ''} {state or ''}".strip()
    guide = LocalGuide(
        property_id=property_id,
        location_name=location_name,
    )

    # ── Step 1: Merge and deduplicate ─────────────────────────────────────
    merged = _merge_places(google_places, yelp_places)
    guide.total_places = len(merged)
    sources = []
    if google_places:
        sources.append(DataSource.GOOGLE_PLACES)
    if yelp_places:
        sources.append(DataSource.YELP_FUSION)
    guide.sources_used = sources

    # ── Step 2: Owner Don't Miss picks ───────────────────────────────────
    dont_miss = _build_dont_miss_picks(owner_dont_miss, merged)
    guide.dont_miss_picks = dont_miss

    # ── Step 3: Vibe-filtered primary recommendations ────────────────────
    primary = _select_primary_recommendations(
        merged, vibe_profile, dont_miss, PRIMARY_RECOMMENDATIONS_COUNT
    )
    guide.primary_recommendations = primary

    # ── Step 4: Organise full category browser ────────────────────────────
    guide.places_by_category = _organise_by_category(merged)

    # ── Step 5: Generate area introduction ───────────────────────────────
    if anthropic_client:
        guide.area_introduction = _generate_area_intro(
            location_name, vibe_profile, primary, dont_miss,
            neighborhood_description, anthropic_client
        )
    else:
        guide.area_introduction = _fallback_area_intro(location_name, vibe_profile)

    logger.info(
        f"[Agent 4] Local guide assembled for property {property_id}. "
        f"Places: {len(merged)}, Primary: {len(primary)}, "
        f"Don't Miss: {len(dont_miss)}"
    )
    return guide


# ── Merge & Deduplication ─────────────────────────────────────────────────

def _merge_places(
    google: list[Place],
    yelp: list[Place],
) -> list[Place]:
    """
    Merge Google and Yelp results into a single deduplicated list.
    When the same business appears in both, merge the records using
    the defined field priority policy.
    """
    merged: list[Place] = []
    yelp_unmatched = list(yelp)

    for g_place in google:
        # Try to find a matching Yelp record
        match_idx = _find_yelp_match(g_place, yelp_unmatched)
        if match_idx is not None:
            y_place = yelp_unmatched.pop(match_idx)
            # Merge: Google is primary, Yelp fills gaps and adds its strengths
            merged_place = _merge_records(g_place, y_place)
        else:
            merged_place = g_place
        merged.append(merged_place)

    # Add remaining unmatched Yelp places
    merged.extend(yelp_unmatched)

    # Sort by composite rating descending, then by review count descending
    merged.sort(
        key=lambda p: (
            -(p.composite_rating() or 0),
            -(p.yelp_review_count or p.google_review_count or 0),
        )
    )

    return merged


def _find_yelp_match(
    google_place: Place,
    yelp_candidates: list[Place],
) -> Optional[int]:
    """
    Find the best-matching Yelp place for a Google place.
    Uses name similarity + distance proximity.
    Returns index into yelp_candidates or None.
    """
    best_idx = None
    best_score = DEDUP_SIMILARITY_THRESHOLD

    g_name = google_place.name.lower().strip()
    g_lat  = google_place.latitude
    g_lng  = google_place.longitude

    for i, y_place in enumerate(yelp_candidates):
        y_name = y_place.name.lower().strip()
        name_sim = SequenceMatcher(None, g_name, y_name).ratio()

        if name_sim < DEDUP_SIMILARITY_THRESHOLD:
            continue

        # If both have coordinates, require proximity (within 0.1 miles)
        if g_lat and g_lng and y_place.latitude and y_place.longitude:
            from agents.agent4.google_places import _haversine_miles
            dist = _haversine_miles(g_lat, g_lng, y_place.latitude, y_place.longitude)
            if dist is not None and dist > 0.15:
                continue

        if name_sim > best_score:
            best_score = name_sim
            best_idx = i

    return best_idx


def _merge_records(google: Place, yelp: Place) -> Place:
    """
    Merge a Google Place and its matching Yelp record.
    Merge policy per TS-11:
      Google: name, address, hours, phone, website
      Yelp:   price_level (if Google missing), review_count (supplement)
      Both:   rating fields preserved separately for composite calculation
    """
    merged = Place(
        name=google.name,                                    # Google wins
        category=google.category,
        primary_source=DataSource.GOOGLE_PLACES,
        address=google.address or yelp.address,
        distance_miles=google.distance_miles or yelp.distance_miles,
        latitude=google.latitude or yelp.latitude,
        longitude=google.longitude or yelp.longitude,
        google_rating=google.google_rating,
        google_review_count=google.google_review_count,
        yelp_rating=yelp.yelp_rating,
        yelp_review_count=yelp.yelp_review_count,
        price_level=google.price_level or yelp.price_level,  # Yelp fills gap
        phone=google.phone or yelp.phone,
        website=google.website or yelp.website,
        photo_url=google.photo_url or yelp.photo_url,
        google_place_id=google.google_place_id,
        yelp_id=yelp.yelp_id,
        hours=google.hours,
    )
    return merged


# ── Don't Miss Picks ──────────────────────────────────────────────────────

def _build_dont_miss_picks(
    owner_picks: list[str],
    all_places: list[Place],
) -> list[DontMissPick]:
    """
    Build the Don't Miss section from owner-provided text.
    Tries to match each pick to an API record for enriched display.
    Owner text is always preserved verbatim as the description.
    """
    picks: list[DontMissPick] = []
    for pick_text in owner_picks[:5]:   # Max 5 Don't Miss picks
        if not pick_text or not pick_text.strip():
            continue

        # Try to match to an API record
        matched_place = _fuzzy_match_pick(pick_text, all_places)

        picks.append(DontMissPick(
            name=_extract_name_from_pick(pick_text),
            description=pick_text.strip(),
            category=matched_place.category if matched_place else None,
            place_ref=matched_place,
        ))

    return picks


def _extract_name_from_pick(pick_text: str) -> str:
    """Extract the place name from an owner pick description."""
    # Take the first sentence or phrase up to a colon/dash if present
    for sep in [":", " - ", " — "]:
        if sep in pick_text:
            return pick_text.split(sep)[0].strip()
    # Otherwise take first few words as the name
    words = pick_text.split()
    return " ".join(words[:4]) if len(words) > 4 else pick_text


def _fuzzy_match_pick(pick_text: str, places: list[Place]) -> Optional[Place]:
    """Try to match an owner's text description to a known Place."""
    pick_lower = pick_text.lower()
    best_match = None
    best_score = 0.7

    for place in places:
        name_lower = place.name.lower()
        # Direct name mention
        if name_lower in pick_lower:
            return place
        # Fuzzy match
        score = SequenceMatcher(None, name_lower, pick_lower[:len(name_lower)+5]).ratio()
        if score > best_score:
            best_score = score
            best_match = place

    return best_match


# ── Vibe Filter ───────────────────────────────────────────────────────────

def _select_primary_recommendations(
    places: list[Place],
    vibe_profile: str,
    dont_miss: list[DontMissPick],
    count: int,
) -> list[Place]:
    """
    Select primary recommendations using vibe priority.
    Excludes places already in Don't Miss section.
    Returns top `count` places across prioritised categories.
    """
    dont_miss_names = {p.name.lower() for p in dont_miss}
    priority_categories = VIBE_PRIMARY_CATEGORIES.get(vibe_profile, list(PlaceCategory))

    # Score each place: base score from rating/reviews + vibe category priority bonus
    scored: list[tuple[float, Place]] = []
    for place in places:
        if place.name.lower() in dont_miss_names:
            continue
        if not place.category:
            continue

        # Base quality score (0-1)
        rating = place.composite_rating() or 3.5
        quality = (rating - 1) / 4   # Normalise 1-5 → 0-1

        # Vibe category bonus
        try:
            cat_rank = priority_categories.index(place.category)
            vibe_bonus = 1.0 - (cat_rank / len(priority_categories))
        except ValueError:
            vibe_bonus = 0.1   # Not in priority list but still available

        score = quality * 0.5 + vibe_bonus * 0.5
        place.vibe_match_score = round(score, 3)
        scored.append((score, place))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    # Select top N ensuring category diversity
    # (no more than 3 per category in the primary section)
    selected: list[Place] = []
    category_counts: dict[PlaceCategory, int] = {}
    for score, place in scored:
        cat_count = category_counts.get(place.category, 0)
        if cat_count >= 3:
            continue
        selected.append(place)
        category_counts[place.category] = cat_count + 1
        if len(selected) >= count:
            break

    # Assign display order
    for i, place in enumerate(selected):
        place.display_order = i + 1

    return selected


# ── Category Browser ──────────────────────────────────────────────────────

def _organise_by_category(places: list[Place]) -> dict[str, list[Place]]:
    """Organise all places into the category browser dict."""
    by_cat: dict[str, list[Place]] = {}
    for place in places:
        cat_key = place.category.value if place.category else "uncategorised"
        by_cat.setdefault(cat_key, []).append(place)
    # Sort each category by composite rating
    for cat in by_cat:
        by_cat[cat].sort(key=lambda p: -(p.composite_rating() or 0))
    return by_cat


# ── Area Introduction ─────────────────────────────────────────────────────

def _generate_area_intro(
    location_name: str,
    vibe_profile: str,
    primary_places: list[Place],
    dont_miss: list[DontMissPick],
    owner_neighborhood_desc: Optional[str],
    anthropic_client: anthropic.Anthropic,
) -> str:
    """
    Generate a 3-5 sentence area introduction using Claude Haiku.
    Written in the property's vibe voice. References real places.
    """
    place_names = [p.name for p in primary_places[:5]]
    dont_miss_names = [p.name for p in dont_miss]

    prompt = f"""Write a 3-5 sentence area introduction for a vacation rental property's local guide section.

Location: {location_name}
Property vibe: {vibe_profile}
Top nearby places: {', '.join(place_names[:5]) if place_names else 'various local spots'}
Owner's highlights: {', '.join(dont_miss_names[:3]) if dont_miss_names else 'see picks below'}
Owner's own description: {owner_neighborhood_desc or 'Not provided'}

Requirements:
- Written in the vibe's voice (romantic = sensory/intimate, family = warm/energetic, etc.)
- Mentions what the area feels like, not just what's there
- References 1-2 real place names naturally
- 3-5 sentences only
- No bullet points, no headers, just flowing prose

Respond with only the introduction text, nothing else."""

    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        logger.error(f"Area intro generation failed: {exc}")
        return _fallback_area_intro(location_name, vibe_profile)


def _fallback_area_intro(location_name: str, vibe_profile: str) -> str:
    """Fallback area introduction when Claude is unavailable."""
    fallbacks = {
        VibeProfile.ROMANTIC_ESCAPE:
            f"{location_name} moves at the pace you set. "
            f"Quiet mornings, unhurried evenings, and the kind of places "
            f"that feel like they were waiting just for you.",
        VibeProfile.FAMILY_ADVENTURE:
            f"{location_name} has everything a family needs for a trip "
            f"they'll talk about for years — adventure, great food, and "
            f"space to simply be together.",
        VibeProfile.MULTIGENERATIONAL:
            f"{location_name} offers something for everyone, from the "
            f"youngest to the wisest in the group. "
            f"Great dining, easy outings, and room for all.",
        VibeProfile.WELLNESS_RETREAT:
            f"{location_name} is the kind of place that invites you "
            f"to slow down. Nature close by, unhurried mornings, "
            f"and spaces designed for genuine rest.",
        VibeProfile.ADVENTURE_BASE_CAMP:
            f"Everything you came for is right outside the door. "
            f"{location_name} is your launching pad — "
            f"gear up, fuel up, and go.",
        VibeProfile.SOCIAL_CELEBRATIONS:
            f"{location_name} knows how to show a group a good time. "
            f"Great restaurants, lively spots, and experiences "
            f"worth celebrating.",
    }
    return fallbacks.get(vibe_profile, f"Explore everything {location_name} has to offer.")
