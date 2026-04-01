"""
TS-21 — Schema.org Structured Data Markup
Tool: Schema.org JSON-LD (no vendor — open standard, $0)

Generates a VacationRental JSON-LD block embedded in the <head>
of every property landing page at build time.

Enables:
  1. Google rich results — price, rating, availability in search results
  2. AI discoverability — Google AI Overviews, Perplexity, travel AI agents
     surface structured property data in AI-generated recommendations.

Must be built at page generation time, not retrofitted later.
At 14,000 properties, retrofitting requires rebuilding every page.

JSON-LD is Google's recommended format (vs Microdata or RDFa).
Does not affect page appearance — machine-readable only.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Schema.org VacationRental type hierarchy
# VacationRental > LodgingBusiness > LocalBusiness > Organization
SCHEMA_TYPE = "VacationRental"


def generate_schema_jsonld(
    name: Optional[str],
    description: Optional[str],
    page_url: str,
    booking_url: Optional[str],
    address_line1: Optional[str],
    city: Optional[str],
    state: Optional[str],
    zip_code: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    bedrooms: Optional[int],
    bathrooms: Optional[float],
    max_occupancy: Optional[int],
    amenities: list[str],
    hero_photo_url: Optional[str],
    google_rating: Optional[float],
    google_review_count: Optional[int],
    avg_nightly_rate: Optional[float],
    slug: str,
) -> str:
    """
    Generate the complete Schema.org JSON-LD block for a VacationRental.
    Returns an HTML <script> tag string for embedding in <head>.

    All fields are optional — the schema degrades gracefully to whatever
    data is available. Missing fields are simply omitted from the output.
    """
    schema: dict = {
        "@context": "https://schema.org",
        "@type": SCHEMA_TYPE,
        "url": page_url,
    }

    # ── Core identity ─────────────────────────────────────────────────────
    if name:
        schema["name"] = name
    if description:
        # Truncate description to 500 chars for the schema — full text is on page
        schema["description"] = description[:500].strip()

    # ── Location ─────────────────────────────────────────────────────────
    address_parts = [
        p for p in [address_line1, city, state, zip_code, "US"]
        if p
    ]
    if city or state:
        schema["address"] = {
            "@type": "PostalAddress",
            "addressLocality": city or "",
            "addressRegion": state or "",
            "postalCode": zip_code or "",
            "addressCountry": "US",
        }
        if address_line1:
            schema["address"]["streetAddress"] = address_line1

    if latitude is not None and longitude is not None:
        schema["geo"] = {
            "@type": "GeoCoordinates",
            "latitude": latitude,
            "longitude": longitude,
        }

    # ── Property specifications ───────────────────────────────────────────
    if bedrooms is not None:
        schema["numberOfRooms"] = bedrooms
        schema["numberOfBedrooms"] = bedrooms

    if bathrooms is not None:
        schema["numberOfBathroomsTotal"] = bathrooms

    if max_occupancy is not None:
        schema["occupancy"] = {
            "@type": "QuantitativeValue",
            "value": max_occupancy,
        }

    # ── Amenities ─────────────────────────────────────────────────────────
    if amenities:
        schema["amenityFeature"] = [
            {
                "@type": "LocationFeatureSpecification",
                "name": amenity,
                "value": True,
            }
            for amenity in amenities[:20]   # Cap at 20 for schema cleanliness
        ]

    # ── Photo ─────────────────────────────────────────────────────────────
    if hero_photo_url:
        schema["image"] = hero_photo_url

    # ── Rating ────────────────────────────────────────────────────────────
    if google_rating and google_review_count:
        schema["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": google_rating,
            "reviewCount": google_review_count,
            "bestRating": 5,
            "worstRating": 1,
        }

    # ── Pricing ───────────────────────────────────────────────────────────
    if avg_nightly_rate:
        schema["priceRange"] = f"From ${int(avg_nightly_rate)}/night"
        schema["offers"] = {
            "@type": "Offer",
            "price": avg_nightly_rate,
            "priceCurrency": "USD",
            "availability": "https://schema.org/InStock",
            "url": booking_url or page_url,
        }

    # ── Booking action ────────────────────────────────────────────────────
    if booking_url:
        schema["potentialAction"] = {
            "@type": "ReserveAction",
            "target": {
                "@type": "EntryPoint",
                "urlTemplate": booking_url,
                "actionPlatform": [
                    "http://schema.org/DesktopWebPlatform",
                    "http://schema.org/MobileWebPlatform",
                ],
            },
            "result": {
                "@type": "LodgingReservation",
                "name": f"Book {name or 'this property'}",
            },
        }

    # ── Serialize to script tag ───────────────────────────────────────────
    json_str = json.dumps(schema, indent=2, ensure_ascii=False)
    return f'<script type="application/ld+json">\n{json_str}\n</script>'


def build_schema_from_inputs(
    kb: dict,
    content_package: dict,
    visual_media: dict,
    page_url: str,
    slug: str,
) -> str:
    """
    Convenience wrapper: extracts all needed fields from Agent outputs
    and calls generate_schema_jsonld.
    """
    def _val(obj, key):
        """Extract .value from a PropertyField dict."""
        f = obj.get(key)
        if isinstance(f, dict):
            return f.get("value")
        return f

    name          = _val(kb, "name")
    description   = content_package.get("property_description") or _val(kb, "description")
    booking_url   = kb.get("booking_url")
    city          = _val(kb, "city")
    state_abbr    = _val(kb, "state")
    zip_code      = _val(kb, "zip_code")
    address_line1 = _val(kb, "address_line1")
    lat           = _val(kb, "latitude")
    lng           = _val(kb, "longitude")
    bedrooms      = _val(kb, "bedrooms")
    bathrooms     = _val(kb, "bathrooms")
    max_occupancy = _val(kb, "max_occupancy")
    avg_rate      = _val(kb, "avg_nightly_rate")
    hero_url      = visual_media.get("hero_photo_url")
    g_rating      = _val(kb, "airbnb_rating")     # Best available rating
    g_review_count = _val(kb, "airbnb_review_count")
    amenities     = [
        a.get("value", "") for a in (kb.get("amenities") or [])[:20]
        if isinstance(a, dict) and a.get("value")
    ]

    return generate_schema_jsonld(
        name=name,
        description=description,
        page_url=page_url,
        booking_url=booking_url,
        address_line1=address_line1,
        city=city,
        state=state_abbr,
        zip_code=zip_code,
        latitude=float(lat) if lat is not None else None,
        longitude=float(lng) if lng is not None else None,
        bedrooms=int(bedrooms) if bedrooms is not None else None,
        bathrooms=float(bathrooms) if bathrooms is not None else None,
        max_occupancy=int(max_occupancy) if max_occupancy is not None else None,
        amenities=amenities,
        hero_photo_url=hero_url,
        google_rating=float(g_rating) if g_rating else None,
        google_review_count=int(g_review_count) if g_review_count else None,
        avg_nightly_rate=float(avg_rate) if avg_rate else None,
        slug=slug,
    )
