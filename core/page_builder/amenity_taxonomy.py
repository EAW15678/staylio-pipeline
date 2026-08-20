"""
Canonical amenity taxonomy — ~188 entries across 8 categories.

Created 2026-08-20 (AMENITY-2). Merged from amenity-taxonomy.md (v1) and
amenity-taxonomy-v2.md. Researched from Airbnb's 591-entry internal system,
VRBO/Expedia category checklists, and 2024-2026 vacation rental award coverage.

This is the single source both repos (staylio-pipeline and staylio-web) draw
from. The wizard's checkbox list and the pipeline's photo-matching logic both
reference these exact names and categories.

Eight categories:
  setting          — fixed geography (ocean views, mountain views, waterfront)
  water_wellness   — pools, hot tubs, saunas, spa
  recreation       — courts, games, gym, fire pit
  gear_transport   — boats, bikes, parking, EV charging
  work_connectivity — WiFi, workspace, streaming
  family_accessibility — cribs, high chairs, wheelchair access
  gathering        — outdoor dining, grills, fireplaces, entertaining
  practical        — kitchen, laundry, HVAC, security, safety

Photo evidence for `setting` items comes from observations.is_setting /
setting_subject, not from located_amenities. All other categories match
against located_amenities within the same category.
"""

# Each category: (key, display_label, [item_names])
AMENITY_CATEGORIES: list[dict] = [
    {
        "key": "setting",
        "label": "Setting",
        "items": [
            "Ocean views", "Mountain views", "Waterfront", "Lake / Waterfront",
            "Beachfront", "Private beach", "Ski-in/Ski-out", "Hot springs access",
        ],
    },
    {
        "key": "water_wellness",
        "label": "Water & Wellness",
        "items": [
            "Private pool", "Shared pool", "Infinity pool", "Saltwater pool",
            "Lap pool", "Plunge pool", "Heated pool", "Indoor pool", "Kiddie pool",
            "Pool with waterslide", "Hot tub", "Private hot tub", "Shared hot tub",
            "Saltwater hot tub", "Sauna", "Infrared sauna", "Steam room",
            "Spa / spa access", "Jetted tub", "Soaking tub", "Outdoor shower",
        ],
    },
    {
        "key": "recreation",
        "label": "Recreation",
        "items": [
            "Tennis court", "Pickleball court", "Basketball court",
            "Volleyball court", "Bocce court", "Croquet", "Horseshoes",
            "Shuffleboard", "Putting green", "Golf course access", "Pool table",
            "Ping pong table", "Foosball table", "Air hockey table", "Arcade games",
            "Board games", "Game room", "Home theater / media room",
            "Projector and screen", "Gym / exercise equipment", "Yoga studio",
            "Trampoline (in-ground or standard)", "Playground / jungle gym",
            "Zip-line", "Climbing wall",
            "Sport court (multi-use)", "Treehouse element", "Barn", "Art studio",
        ],
    },
    {
        "key": "gear_transport",
        "label": "Gear & Transport",
        "items": [
            "Kayak", "Canoe", "Paddleboard", "Sailboat", "Jet skis", "Boat",
            "Fishing gear", "Snorkeling equipment", "Beach chairs",
            "Beach umbrellas", "Water skis", "Bikes", "Golf cart", "ATV / UTV",
            "EV charger", "Free parking on premises", "Garage parking",
            "Driveway parking", "Street parking", "Covered parking",
        ],
    },
    {
        "key": "work_connectivity",
        "label": "Work & Connectivity",
        "items": [
            "High-speed WiFi", "Dedicated workspace", "Ethernet connection",
            "Printer", "Smart TV with streaming", "Cell signal booster",
        ],
    },
    {
        "key": "family_accessibility",
        "label": "Family & Accessibility",
        "items": [
            "Crib", "Pack 'n play", "High chair", "Baby gate", "Baby monitor",
            "Changing table", "Children's books and toys", "Board games for kids",
            "Family/kid-friendly",
            "Wheelchair accessible", "Step-free guest entrance", "Wide doorways",
            "Grab bars (shower/toilet)", "Roll-in shower", "Accessible-height bed",
            "Accessible parking spot", "Single-level home",
        ],
    },
    {
        "key": "gathering",
        "label": "Gathering",
        "items": [
            "Outdoor dining area", "Fire pit", "BBQ grill (gas)",
            "BBQ grill (charcoal)", "Outdoor kitchen", "Bar / wet bar",
            "Wine cellar / wine fridge", "Hammock", "Gazebo", "Pergola",
            "Deck / patio / balcony", "Rooftop deck", "Dining table (seats 8+)",
            "Fireplace (indoor)", "Fireplace (outdoor)",
        ],
    },
    {
        "key": "practical",
        "label": "Practical",
        "items": [
            "Full kitchen", "Kitchenette", "Refrigerator", "Dishwasher",
            "Microwave", "Coffee maker", "Washer", "Dryer", "Iron",
            "Air conditioning (central)", "Air conditioning (window unit)",
            "Heating", "Ceiling fans", "Self check-in / smart lock",
            "Keyless entry", "Security cameras (exterior only, disclosed)",
            "Smoke alarm", "Carbon monoxide alarm", "Fire extinguisher",
            "First aid kit", "Safe", "Elevator", "Generator", "Solar power",
            "Pet-friendly", "Smart home technology", "Chef's kitchen",
            "Smoking allowed (exterior only)",
        ],
    },
]

# Flat lookup: amenity name → category key
AMENITY_TO_CATEGORY: dict[str, str] = {}
for _cat in AMENITY_CATEGORIES:
    for _item in _cat["items"]:
        AMENITY_TO_CATEGORY[_item] = _cat["key"]

# All valid amenity names
ALL_AMENITY_NAMES: frozenset = frozenset(AMENITY_TO_CATEGORY.keys())

# Setting category items — these match via is_setting/setting_subject
SETTING_AMENITIES: frozenset = frozenset(
    item for cat in AMENITY_CATEGORIES if cat["key"] == "setting"
    for item in cat["items"]
)

# Synonym map for matching amenity names against located_amenities[].name
# Used for the 7 non-setting categories. Each amenity name maps to a set of
# substrings that match observation located_amenities[].name values.
# Category narrows the candidate pool; name decides precision.
_AMENITY_PHOTO_SYNONYMS: dict[str, set] = {
    # water_wellness
    "Private pool": {"pool", "swimming pool"},
    "Shared pool": {"pool", "swimming pool"},
    "Infinity pool": {"infinity pool", "pool"},
    "Saltwater pool": {"pool"},
    "Lap pool": {"lap pool", "pool"},
    "Plunge pool": {"plunge pool", "pool"},
    "Heated pool": {"pool"},
    "Indoor pool": {"indoor pool", "pool"},
    "Kiddie pool": {"kiddie pool"},
    "Pool with waterslide": {"waterslide", "water slide"},
    "Hot tub": {"hot tub", "spa", "jacuzzi"},
    "Private hot tub": {"hot tub", "jacuzzi"},
    "Shared hot tub": {"hot tub"},
    "Saltwater hot tub": {"hot tub"},
    "Sauna": {"sauna"},
    "Infrared sauna": {"sauna"},
    "Steam room": {"steam room"},
    "Spa / spa access": {"spa"},
    "Jetted tub": {"jetted tub", "jacuzzi"},
    "Soaking tub": {"soaking tub", "bathtub"},
    "Outdoor shower": {"outdoor shower"},
    # recreation
    "Tennis court": {"tennis"},
    "Pickleball court": {"pickleball"},
    "Basketball court": {"basketball"},
    "Volleyball court": {"volleyball"},
    "Bocce court": {"bocce"},
    "Putting green": {"putting green"},
    "Golf course access": {"golf"},
    "Pool table": {"pool table", "billiard"},
    "Ping pong table": {"ping pong", "table tennis"},
    "Foosball table": {"foosball"},
    "Air hockey table": {"air hockey"},
    "Arcade games": {"arcade"},
    "Game room": {"game room"},
    "Home theater / media room": {"theater", "theatre", "media room"},
    "Projector and screen": {"projector"},
    "Gym / exercise equipment": {"gym", "exercise", "fitness"},
    "Yoga studio": {"yoga"},
    "Trampoline (in-ground or standard)": {"trampoline"},
    "Playground / jungle gym": {"playground", "jungle gym"},
    "Fire pit": {"fire pit"},
    "Treehouse element": {"treehouse"},
    "Barn": {"barn"},
    "Art studio": {"art studio"},
    # gear_transport
    "Kayak": {"kayak"},
    "Canoe": {"canoe"},
    "Paddleboard": {"paddleboard"},
    "Bikes": {"bike", "bicycle"},
    "Golf cart": {"golf cart"},
    "EV charger": {"ev charger", "charging station"},
    "Fishing gear": {"fishing"},
    "Covered parking": {"covered parking", "carport"},
    "Garage parking": {"garage"},
    # work_connectivity
    "Dedicated workspace": {"desk", "workspace", "office"},
    # family_accessibility
    "Wheelchair accessible": {"wheelchair", "accessible"},
    "Step-free guest entrance": {"step-free", "ramp"},
    "Grab bars (shower/toilet)": {"grab bar"},
    "Roll-in shower": {"roll-in shower"},
    "High chair": {"high chair"},
    "Crib": {"crib"},
    # gathering
    "Outdoor dining area": {"outdoor dining", "dining table"},
    "BBQ grill (gas)": {"grill", "bbq", "barbecue"},
    "BBQ grill (charcoal)": {"grill", "bbq", "barbecue", "charcoal"},
    "Outdoor kitchen": {"outdoor kitchen"},
    "Bar / wet bar": {"bar", "wet bar"},
    "Wine cellar / wine fridge": {"wine cellar", "wine fridge"},
    "Hammock": {"hammock"},
    "Gazebo": {"gazebo"},
    "Pergola": {"pergola"},
    "Deck / patio / balcony": {"deck", "patio", "balcony"},
    "Rooftop deck": {"rooftop"},
    "Dining table (seats 8+)": {"dining table"},
    "Fireplace (indoor)": {"fireplace"},
    "Fireplace (outdoor)": {"fireplace", "fire pit"},
    # practical
    "Full kitchen": {"kitchen"},
    "Coffee maker": {"coffee"},
    "Washer": {"washer", "washing machine"},
    "Dryer": {"dryer"},
    "Elevator": {"elevator"},
    "Solar power": {"solar"},
}

# Setting synonyms — matched against setting_subject text, not located_amenities
_SETTING_PHOTO_SYNONYMS: dict[str, set] = {
    "Ocean views": {"ocean", "sea", "atlantic", "pacific", "beach"},
    "Mountain views": {"mountain", "peak", "ridge", "alpine"},
    "Waterfront": {"waterfront", "water view", "waterway", "lake", "river"},
    "Lake / Waterfront": {"lake", "waterfront", "water view"},
    "Beachfront": {"beach", "ocean", "shore", "coast"},
    "Private beach": {"beach", "private beach"},
    "Ski-in/Ski-out": {"ski"},
    "Hot springs access": {"hot spring"},
}


def extract_amenity_names(amenities_list: list) -> list:
    """Extract just the name strings from a structured amenities list.

    Handles both old format (list[str]) and new format (list[{name, category}]).
    Used by skills that only need the names for prompts, not the categories.
    """
    names = []
    for a in (amenities_list or []):
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, dict) and a.get("name"):
            names.append(a["name"])
    return names
