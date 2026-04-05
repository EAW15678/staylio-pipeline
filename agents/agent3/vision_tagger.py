"""
TS-07b — Photo Library Scoring, Tagging & Hero Selection
Tool: Google Cloud Vision API

Runs all 100 photos through Vision API after Claid.ai enhancement.
Returns structured metadata for the permanent media library.

Two-stage hero selection:
  Stage 1: Identify category winners (best photo per subject category
            by composition score)
  Stage 2: Rank category winners by vibe-profile priority to select
            the final hero image

Provenance consistency check: compares Vision API labels between
original and enhanced versions. If labels diverge materially,
the photo is flagged — this is the automated enforcement layer
for the Claid.ai operation whitelist.
"""

import json
import os
import logging
from typing import Optional

from google.cloud import vision
from google.cloud.vision_v1 import types
from google.oauth2 import service_account

from agents.agent3.models import (
    MediaAsset,
    SubjectCategory,
)
from models.property import VibeProfile

logger = logging.getLogger(__name__)

GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")


def _get_vision_client() -> vision.ImageAnnotatorClient:
    """
    Return a Vision API client.
    If GOOGLE_SERVICE_ACCOUNT_JSON is set, use it as inline service account JSON.
    Otherwise fall back to Application Default Credentials.
    """
    if GOOGLE_SERVICE_ACCOUNT_JSON:
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return vision.ImageAnnotatorClient(credentials=creds)
    return vision.ImageAnnotatorClient()


# Label divergence threshold for provenance flag
# If the Jaccard similarity between original and enhanced label sets
# falls below this, the photo is flagged for human review.
PROVENANCE_SIMILARITY_THRESHOLD = 0.6


# ── Label → SubjectCategory mapping ──────────────────────────────────────
# Vision API returns natural language labels. We map them to our taxonomy.
# This is the label mapping required by TS-07b before Agent 3 build.

LABEL_TO_CATEGORY: dict[str, SubjectCategory] = {
    # Exterior
    "facade": SubjectCategory.EXTERIOR,
    "house": SubjectCategory.EXTERIOR,
    "building": SubjectCategory.EXTERIOR,
    "villa": SubjectCategory.EXTERIOR,
    "cabin": SubjectCategory.EXTERIOR,
    "cottage": SubjectCategory.EXTERIOR,
    "exterior": SubjectCategory.EXTERIOR,
    "porch": SubjectCategory.EXTERIOR,
    "driveway": SubjectCategory.EXTERIOR,

    # Living areas
    "living room": SubjectCategory.LIVING_ROOM,
    "couch": SubjectCategory.LIVING_ROOM,
    "sofa": SubjectCategory.LIVING_ROOM,
    "fireplace": SubjectCategory.LIVING_ROOM,
    "family room": SubjectCategory.LIVING_ROOM,
    "great room": SubjectCategory.LIVING_ROOM,

    # Kitchen
    "kitchen": SubjectCategory.KITCHEN,
    "countertop": SubjectCategory.KITCHEN,
    "stove": SubjectCategory.KITCHEN,
    "refrigerator": SubjectCategory.KITCHEN,
    "dining room": SubjectCategory.KITCHEN,
    "dining table": SubjectCategory.KITCHEN,

    # Bedrooms
    "bedroom": SubjectCategory.STANDARD_BEDROOM,
    "bed": SubjectCategory.STANDARD_BEDROOM,
    "master bedroom": SubjectCategory.MASTER_BEDROOM,

    # Bathrooms
    "bathroom": SubjectCategory.BATHROOM,
    "bathtub": SubjectCategory.BATHROOM,
    "shower": SubjectCategory.BATHROOM,
    "sink": SubjectCategory.BATHROOM,

    # Pool / Hot Tub
    "swimming pool": SubjectCategory.POOL_HOT_TUB,
    "pool": SubjectCategory.POOL_HOT_TUB,
    "hot tub": SubjectCategory.POOL_HOT_TUB,
    "jacuzzi": SubjectCategory.POOL_HOT_TUB,
    "spa": SubjectCategory.POOL_HOT_TUB,

    # Outdoor entertaining
    "patio": SubjectCategory.OUTDOOR_ENTERTAINING,
    "deck": SubjectCategory.OUTDOOR_ENTERTAINING,
    "balcony": SubjectCategory.OUTDOOR_ENTERTAINING,
    "terrace": SubjectCategory.OUTDOOR_ENTERTAINING,
    "outdoor kitchen": SubjectCategory.OUTDOOR_ENTERTAINING,
    "fire pit": SubjectCategory.OUTDOOR_ENTERTAINING,
    "bbq": SubjectCategory.OUTDOOR_ENTERTAINING,

    # Views
    "ocean": SubjectCategory.VIEW,
    "sea": SubjectCategory.VIEW,
    "lake": SubjectCategory.VIEW,
    "mountain": SubjectCategory.VIEW,
    "sunset": SubjectCategory.VIEW,
    "sunrise": SubjectCategory.VIEW,
    "water": SubjectCategory.VIEW,
    "horizon": SubjectCategory.VIEW,
    "landscape": SubjectCategory.VIEW,
    "nature": SubjectCategory.VIEW,
    "forest": SubjectCategory.VIEW,
    "beach": SubjectCategory.VIEW,

    # Entertainment
    "game room": SubjectCategory.GAME_ENTERTAINMENT,
    "billiards": SubjectCategory.GAME_ENTERTAINMENT,
    "pool table": SubjectCategory.GAME_ENTERTAINMENT,
    "home theater": SubjectCategory.GAME_ENTERTAINMENT,
    "television": SubjectCategory.GAME_ENTERTAINMENT,
    "arcade": SubjectCategory.GAME_ENTERTAINMENT,
}


# ── Vibe → Category Priority ──────────────────────────────────────────────
# For hero selection Stage 2: given a vibe profile, which subject
# categories take priority when selecting the final hero image.
# First category in the list that has a winner becomes the hero.

VIBE_HERO_PRIORITY: dict[str, list[SubjectCategory]] = {
    VibeProfile.ROMANTIC_ESCAPE: [
        SubjectCategory.VIEW,
        SubjectCategory.MASTER_BEDROOM,
        SubjectCategory.POOL_HOT_TUB,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.EXTERIOR,
    ],
    VibeProfile.FAMILY_ADVENTURE: [
        SubjectCategory.POOL_HOT_TUB,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.EXTERIOR,
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.VIEW,
        SubjectCategory.GAME_ENTERTAINMENT,
    ],
    VibeProfile.MULTIGENERATIONAL: [
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.EXTERIOR,
        SubjectCategory.POOL_HOT_TUB,
        SubjectCategory.VIEW,
        SubjectCategory.KITCHEN,
    ],
    VibeProfile.WELLNESS_RETREAT: [
        SubjectCategory.VIEW,
        SubjectCategory.POOL_HOT_TUB,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.MASTER_BEDROOM,
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.EXTERIOR,
    ],
    VibeProfile.ADVENTURE_BASE_CAMP: [
        SubjectCategory.EXTERIOR,
        SubjectCategory.VIEW,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.POOL_HOT_TUB,
    ],
    VibeProfile.SOCIAL_CELEBRATIONS: [
        SubjectCategory.POOL_HOT_TUB,
        SubjectCategory.OUTDOOR_ENTERTAINING,
        SubjectCategory.VIEW,
        SubjectCategory.LIVING_ROOM,
        SubjectCategory.EXTERIOR,
        SubjectCategory.GAME_ENTERTAINMENT,
    ],
}


def tag_and_score_photos(
    assets: list[MediaAsset],
    vibe_profile: str,
    property_id: str,
) -> tuple[list[MediaAsset], Optional[str]]:
    """
    Main Vision API tagging pass.

    For each MediaAsset with an enhanced URL:
      1. Run Vision API label detection + image properties + safe search
      2. Calculate composition score from Vision signals
      3. Map labels to subject category
      4. Run provenance consistency check (vs original labels if available)

    Then:
      5. Select category winners (best photo per category by score)
      6. Select hero from category winners using vibe priority
      7. Queue category winners for social crops

    Returns:
      (updated_assets_list, hero_photo_url)
    """
    client = _get_vision_client()

    for asset in assets:
        url_to_tag = asset.asset_url_enhanced or asset.asset_url_original
        if not url_to_tag:
            continue
        _tag_single_asset(client, asset, url_to_tag)

    # Run provenance check for assets that have both original and enhanced tagged
    _run_provenance_checks(assets)

    # Category winner selection
    category_winners = _select_category_winners(assets)

    # Mark social crop queue
    winner_urls = {url for url in category_winners.values()}
    for asset in assets:
        if asset.asset_url_enhanced in winner_urls:
            asset.social_crop_queued = True

    # Hero selection
    hero_url = _select_hero(category_winners, vibe_profile)

    # Set hero_rank on the winning asset
    if hero_url:
        for asset in assets:
            if asset.asset_url_enhanced == hero_url:
                asset.hero_rank = 1
                break

    logger.info(
        f"[TS-07b] Tagging complete for property {property_id}. "
        f"Assets: {len(assets)}, Category winners: {len(category_winners)}, "
        f"Hero: {hero_url or 'none'}"
    )
    return assets, hero_url


def tag_original_for_provenance(
    asset: MediaAsset,
) -> MediaAsset:
    """
    Tag the ORIGINAL (pre-enhancement) version of a photo to
    establish the baseline for provenance consistency checking.
    Called on originals before Claid.ai enhancement begins.
    """
    if not asset.asset_url_original:
        return asset
    client = _get_vision_client()
    labels = _get_labels(client, asset.asset_url_original)
    asset.labels_original = labels
    return asset


# ── Internal functions ────────────────────────────────────────────────────

def _tag_single_asset(
    client: vision.ImageAnnotatorClient,
    asset: MediaAsset,
    url: str,
) -> None:
    """Run Vision API on a single photo and populate asset fields."""
    try:
        image = types.Image()
        image.source.image_uri = url

        features = [
            types.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
            types.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            types.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
        ]

        response = client.annotate_image({"image": image, "features": features})

        # Labels
        labels = [l.description.lower() for l in response.label_annotations]
        asset.labels_enhanced = labels

        # Subject category
        asset.subject_category = _classify_category(labels)

        # Composition score from image properties
        if response.image_properties_annotation:
            props = response.image_properties_annotation
            brightness, saturation = _extract_dominant_color_stats(props)
            asset.brightness = brightness
            asset.dominant_colors = [
                _color_hex(c.color)
                for c in props.dominant_colors.colors[:5]
            ]
        else:
            brightness = 0.5

        # Sharpness proxy — Vision doesn't give sharpness directly;
        # we use label confidence scores as a proxy for image quality
        if response.label_annotations:
            avg_confidence = sum(l.score for l in response.label_annotations[:5]) / min(5, len(response.label_annotations))
            asset.sharpness = avg_confidence
        else:
            asset.sharpness = 0.5

        # Composite composition score
        asset.composition_score = _compute_composition_score(
            brightness=asset.brightness,
            sharpness=asset.sharpness,
            label_count=len(labels),
        )

        # Safe search
        if response.safe_search_annotation:
            ss = response.safe_search_annotation
            # Fail on likely or very_likely adult/violence content
            likelihood_bad = {
                vision.Likelihood.LIKELY,
                vision.Likelihood.VERY_LIKELY,
            }
            asset.safe_search_pass = (
                ss.adult not in likelihood_bad
                and ss.violence not in likelihood_bad
                and ss.racy not in likelihood_bad
            )

    except Exception as exc:
        logger.error(f"Vision API tagging failed for {url}: {exc}")


def _get_labels(
    client: vision.ImageAnnotatorClient,
    url: str,
) -> list[str]:
    """Get label list for a URL — used for provenance baseline."""
    try:
        image = types.Image()
        image.source.image_uri = url
        resp = client.label_detection(image=image, max_results=20)
        return [l.description.lower() for l in resp.label_annotations]
    except Exception as exc:
        logger.warning(f"Vision labels failed for {url}: {exc}")
        return []


def _run_provenance_checks(assets: list[MediaAsset]) -> None:
    """
    Compare original vs enhanced label sets.
    Flag if Jaccard similarity < PROVENANCE_SIMILARITY_THRESHOLD.
    This detects if the enhancement operation materially changed
    what subjects are visible — which would indicate a prohibited
    operation was applied (e.g. background replacement, virtual staging).
    """
    for asset in assets:
        if not asset.labels_original or not asset.labels_enhanced:
            continue
        orig_set  = set(asset.labels_original)
        enh_set   = set(asset.labels_enhanced)
        if not orig_set and not enh_set:
            continue
        intersection = orig_set & enh_set
        union        = orig_set | enh_set
        jaccard      = len(intersection) / len(union) if union else 1.0
        if jaccard < PROVENANCE_SIMILARITY_THRESHOLD:
            asset.provenance_flag = True
            logger.warning(
                f"[TS-07b] Provenance flag: {asset.asset_url_enhanced} "
                f"Jaccard={jaccard:.2f} — labels diverged between original and enhanced. "
                f"New labels: {enh_set - orig_set}. Missing labels: {orig_set - enh_set}"
            )


def _classify_category(labels: list[str]) -> SubjectCategory:
    """Map Vision API labels to the closest SubjectCategory."""
    # Count matches per category
    category_scores: dict[SubjectCategory, int] = {}
    for label in labels:
        cat = LABEL_TO_CATEGORY.get(label.lower())
        if cat:
            category_scores[cat] = category_scores.get(cat, 0) + 1

    if not category_scores:
        return SubjectCategory.UNCATEGORISED

    # Return the category with the most label matches
    return max(category_scores, key=lambda c: category_scores[c])


def _select_category_winners(
    assets: list[MediaAsset],
) -> dict[SubjectCategory, str]:
    """
    For each subject category, select the photo with the highest
    composition score. Returns {category: enhanced_url}.
    Only includes categories that passed safe search.
    """
    by_category: dict[SubjectCategory, list[MediaAsset]] = {}
    for asset in assets:
        if not asset.safe_search_pass:
            continue
        if asset.provenance_flag:
            continue   # Flagged photos don't enter the library
        cat = asset.subject_category
        if cat == SubjectCategory.UNCATEGORISED:
            continue
        by_category.setdefault(cat, []).append(asset)

    winners: dict[SubjectCategory, str] = {}
    for cat, cat_assets in by_category.items():
        best = max(cat_assets, key=lambda a: a.composition_score)
        # Set category_rank for all assets in this category
        sorted_assets = sorted(cat_assets, key=lambda a: a.composition_score, reverse=True)
        for rank, a in enumerate(sorted_assets, start=1):
            a.category_rank = rank
        url = best.asset_url_enhanced or best.asset_url_original
        if url:
            winners[cat] = url

    return winners


def _select_hero(
    category_winners: dict[SubjectCategory, str],
    vibe_profile: str,
) -> Optional[str]:
    """
    Stage 2 hero selection: pick from category winners using
    vibe-profile priority order.
    Returns the URL of the hero image or None if no winners available.
    """
    priority = VIBE_HERO_PRIORITY.get(vibe_profile, [])
    for category in priority:
        if category in category_winners:
            return category_winners[category]

    # Fallback: any category winner if vibe priority yields nothing
    if category_winners:
        return next(iter(category_winners.values()))
    return None


def _compute_composition_score(
    brightness: float,
    sharpness: float,
    label_count: int,
) -> float:
    """
    Composite quality score 0.0–1.0.
    Weights: sharpness 50%, brightness quality 30%, label richness 20%.
    Brightness quality peaks at 0.55 (slightly bright, well-lit).
    """
    # Brightness quality — penalise overexposed (>0.8) and underexposed (<0.2)
    b_quality = 1.0 - abs(brightness - 0.55) / 0.55
    b_quality = max(0.0, min(1.0, b_quality))

    # Label richness — more meaningful labels = richer, more identifiable scene
    label_quality = min(1.0, label_count / 15.0)

    return (sharpness * 0.5) + (b_quality * 0.3) + (label_quality * 0.2)


def _extract_dominant_color_stats(props) -> tuple[float, float]:
    """Extract average brightness and saturation from dominant colors."""
    if not props.dominant_colors.colors:
        return 0.5, 0.5
    total_brightness = 0.0
    total_saturation = 0.0
    count = 0
    for color_info in props.dominant_colors.colors[:5]:
        c = color_info.color
        r, g, b = c.red / 255.0, c.green / 255.0, c.blue / 255.0
        brightness = (max(r, g, b) + min(r, g, b)) / 2
        saturation = (
            (max(r, g, b) - min(r, g, b)) / (1 - abs(2 * brightness - 1))
            if brightness not in (0, 1) else 0
        )
        total_brightness += brightness
        total_saturation += saturation
        count += 1
    return total_brightness / count, total_saturation / count


def _color_hex(color) -> str:
    return f"#{int(color.red):02x}{int(color.green):02x}{int(color.blue):02x}"
