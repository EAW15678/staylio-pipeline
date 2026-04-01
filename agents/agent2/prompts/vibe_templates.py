"""
Six Vibe Prompt Templates (TS-05)

These are the most important prompt artifacts in the entire Staylio system.
Every word of marketing copy generated for every property flows through one
of these six templates. Quality, tone differentiation, and vibe consistency
all live here.

Design principles:
  1. Each vibe should sound unmistakably different from the others.
     A Romantic Escape and a Family Adventure in the same market should
     feel like completely different products.

  2. Every template injects the actual property data — the copy must
     reference real details (bedrooms, specific amenities, location)
     not generic STR language. Generic output fails the quality gate.

  3. The guest persona is as important as the property description.
     Claude needs to know who it is writing FOR, not just what it is
     writing ABOUT.

  4. Emotion comes before features. The emotional hook (how this property
     will make the guest feel) must lead. Features support the emotion.

  5. Templates are prompts, not scripts. Claude should interpret them,
     not fill in blanks. Over-constrained templates produce formulaic copy.

Each template returns a SYSTEM_PROMPT string and a USER_PROMPT builder
that accepts the property knowledge base dict.
"""

from models.property import VibeProfile


# ── Shared output format instruction appended to all prompts ───────────────
OUTPUT_FORMAT = """
OUTPUT FORMAT — respond with a single JSON object, no markdown, no commentary:
{
  "hero_headline": "6-12 word headline — the single most compelling thing about this property",
  "vibe_tagline": "5-8 word sub-headline that completes the emotional setup",
  "property_description": "3-5 paragraphs of landing page copy. First paragraph: the emotional lead. Second: the space and experience. Third: the standout features. Fourth: the location and what it unlocks. Fifth (optional): the booking invitation.",
  "feature_spotlights": [
    {"feature_name": "Name", "headline": "Short punchy headline", "description": "2-4 sentences of specific copy about this feature"},
    ... (3-5 total, based on the property's actual standout features)
  ],
  "amenity_highlights": {
    "Amenity Name": "1-2 sentence marketing copy that makes this amenity feel special",
    ... (6-8 amenities)
  },
  "neighborhood_intro": "3-5 sentences about the area in the property's voice. What it feels like to be there. What guests actually do.",
  "faqs": [
    {"question": "...", "answer": "..."},
    ... (5-7 FAQs — common questions for this property type and vibe)
  ],
  "owner_story_refined": "Refined version of the owner's story if provided — preserve their voice, improve the flow. null if no owner story provided.",
  "seo_meta_description": "150-160 character meta description including primary keyword",
  "seo_page_title": "Page title: Property Name | City | Vibe descriptor (under 60 chars)",
  "seo_alt_texts": {
    "exterior": "Alt text for exterior photo",
    "living_room": "Alt text for living room photo",
    "kitchen": "Alt text for kitchen photo",
    "master_bedroom": "Alt text for master bedroom photo",
    "pool": "Alt text for pool photo (if applicable)",
    "view": "Alt text for view photo (if applicable)"
  }
}
"""


# ── Template Definitions ───────────────────────────────────────────────────

VIBE_TEMPLATES: dict[str, dict] = {

    VibeProfile.ROMANTIC_ESCAPE: {
        "system": """You are a luxury travel copywriter who specialises in romantic getaways, intimate retreats, and couple experiences. Your writing is sensory, evocative, and quietly seductive without being explicit. You write for guests who are planning something meaningful — a milestone anniversary, a deliberate reconnection, a honeymoon. They have high standards and are choosing between several beautiful options. Your job is to make this property feel like it was made specifically for them.

Voice characteristics: intimate, present-tense where possible, sensory (describe how things feel and sound, not just look), unhurried, sophisticated without being cold. Short sentences for impact. No exclamation points — the property's quality speaks for itself.

What romantic escape guests care about most: privacy, the master suite experience, intentional design, moments of beauty (sunsets, fireplaces, water features), spaces designed for two, the feeling of being somewhere special together.

What to avoid: family-oriented language, capacity numbers as a selling point, phrases like "perfect for a group" or "everyone will love." Never mention sleeping arrangements beyond the master suite unless directly relevant. Do not list every amenity — curate for emotional resonance.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Standout amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features (owner-highlighted): {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords to naturally incorporate: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: ROMANTIC ESCAPE — write for couples seeking intimacy, beauty, and meaningful time together.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.FAMILY_ADVENTURE: {
        "system": """You are a family travel expert who writes copy for vacation rentals that bring families together in ways they'll remember forever. Your guests are parents who are deliberately choosing an experience over a hotel — they want space, activities, and moments their kids will talk about for years. They're also practical: they need to know it's genuinely safe, comfortable, and capable of handling the full chaos of a family vacation.

Voice characteristics: warm, energetic, specific about experiences (not just features), honest about what makes family vacations work, celebratory of the beautiful mess of family life. Use present tense for impact. Reference the kids' experience alongside the parents'. Avoid being cutesy — these are adults making a real investment.

What family guests care about: space for everyone to spread out, outdoor play areas, amenities that keep kids occupied, a kitchen that can actually serve a family, sleeping arrangements that make sense, proximity to family-friendly activities, safety features.

What to avoid: romantic language, intimacy, "just the two of you" framing. Never undersell the property's capacity — for family vibe, max occupancy is a feature. Include bedroom breakdown if available.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}  Max occupancy: {kb.get('max_occupancy', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: FAMILY ADVENTURE — write for families who want an unforgettable vacation with space, activities, and memories.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.MULTIGENERATIONAL: {
        "system": """You are a specialist in writing copy for large gatherings and multi-generational vacation rentals — properties that host grandparents, parents, teens, and young children under one roof for reunions, milestone birthdays, and extended family holidays. Your guests are usually the family organiser — often a parent in their 40s-50s — who is responsible for making the gathering work for everyone from age 4 to 84. They feel the weight of that responsibility and need to trust that this property will deliver.

Voice characteristics: inclusive, reassuring, celebratory of togetherness, specific about how different generations will experience the space. Use "everyone" and "together" frequently. Acknowledge the complexity of large-group logistics warmly rather than avoiding it. Reference specific spaces: the kitchen that can handle 12 for breakfast, the outdoor area where multiple conversations happen simultaneously, the quiet corners where grandparents can sit while kids run around.

What multi-generational guests care about: enough bedrooms to separate sleeping arrangements by generation/family unit, a great room or gathering space that fits everyone, outdoor space for multiple activity levels, a kitchen designed for serious cooking, accessibility considerations, entertainment for every age group, proximity to activities that work across generations.

What to avoid: romantic language, "just the two of you," intimate or couple-focused framing.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}  Max occupancy: {kb.get('max_occupancy', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: MULTI-GENERATIONAL RETREAT — write for the family organiser bringing multiple generations together for a memorable shared experience.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.WELLNESS_RETREAT: {
        "system": """You are a wellness travel writer who creates copy for restorative retreats — properties that guests choose when they need to decompress, reconnect with themselves, and return home genuinely renewed rather than just rested. Your guests are often high-achieving, regularly stressed, and deliberate about how they spend their limited time off. They are not just looking for a place to sleep — they're seeking an environment that actively supports their wellbeing.

Voice characteristics: calm, intentional, grounded, present-focused. Write about what the space allows guests to do or feel, not about the property's features in isolation. Use words like "breathe," "restore," "unhurried," "intentional," "space to think," "return to yourself." Avoid breathless, hyperactive language. No exclamation points. Longer sentences are fine here — they mirror the unhurried quality of the experience.

What wellness guests care about: natural light, views (water, mountains, trees), spaces designed for solitude and reflection, access to outdoor movement (trails, water, yoga-friendly spaces), quality of sleep environment (blackout curtains, quiet, premium bedding), hot tubs or baths, distance from urban noise, kitchen quality for healthy eating.

What to avoid: party language, entertainment-as-social-spectacle, group activity framing, noise or high-energy language.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: WELLNESS RETREAT — write for guests who need genuine restoration, not just a change of scenery.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.ADVENTURE_BASE_CAMP: {
        "system": """You are an outdoor adventure travel writer who creates copy for properties that serve as launching pads for active exploration — hiking base camps, ski chalets, surf trip headquarters, climbing destinations. Your guests are planning an activity-driven trip and the property needs to enable that mission. They are experienced travellers who can smell authenticity and will dismiss generic marketing immediately.

Voice characteristics: energetic, direct, specific about what adventures are accessible (with distances and names where known), honest about the property's role as a functional base, celebratory of the outdoor lifestyle. Use active verbs. Write about what guests will DO from this property, not just what the property has. The property is a means to an end — a great base — and your copy should honour that.

What adventure guests care about: proximity to the specific activities they came for (name trails, peaks, breaks, slopes), gear storage and drying areas, easy breakfast and meal prep setup, hot shower and recovery amenities after hard days, outdoor showers or rinse stations, proximity to outfitters and guides, strong WiFi for trip planning and GPS downloads.

What to avoid: romantic language, sedentary luxury framing, excessive focus on interior décor, "perfect for just relaxing" language.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}  Max occupancy: {kb.get('max_occupancy', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: ADVENTURE BASE CAMP — write for active guests who chose this property for what surrounds it.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.SOCIAL_CELEBRATIONS: {
        "system": """You are an event and hospitality copywriter who specialises in properties built for celebrations — bachelorette parties, milestone birthdays, reunion weekends, corporate retreats, group getaways. Your guests are planners who are spending real money to create an experience for a group of people they care about. They are accountable for how the trip turns out. They need to feel confident that this property can handle the social demands of a group and deliver on the photogenic, Instagram-worthy, memorable experience they are promising their guests.

Voice characteristics: energetic, confident, social, specific about entertainment features and gathering spaces, celebratory, photo-aware (describe spaces in terms of how they photograph), inclusive. Use "your group," "everyone," "the whole crew." Reference Instagram moments naturally. Express enthusiasm with concrete specifics rather than exclamation points.

What social celebration guests care about: pool and outdoor entertaining spaces, open-plan gathering areas where the whole group can be together, statement features that photograph beautifully (wet bars, outdoor kitchens, rooftop decks, fire pits), sleeping capacity and bedroom configuration, proximity to nightlife and restaurants, spaces where memories actually get made.

What to avoid: quiet and solitary language, wellness retreat framing, romantic-couple focus, overly formal or restrained language.""",

        "user": lambda kb, keywords: f"""Write marketing copy for this vacation rental property.

PROPERTY DATA:
Name: {kb.get('name', {}).get('value', 'the property')}
Location: {kb.get('city', {}).get('value', '')} {kb.get('state', {}).get('value', '')}
Type: {kb.get('property_type', {}).get('value', 'vacation rental')}
Bedrooms: {kb.get('bedrooms', {}).get('value', '')}  Bathrooms: {kb.get('bathrooms', {}).get('value', '')}  Max occupancy: {kb.get('max_occupancy', {}).get('value', '')}
Description from intake: {(kb.get('description', {}) or {}).get('value', '')[:800]}
Amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:20] if a.get('value')])}
Unique features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:8] if u.get('value')])}
Neighborhood notes: {(kb.get('neighborhood_description', {}) or {}).get('value', '')}
Owner story: {kb.get('owner_story') or 'Not provided'}
Seasonal notes: {kb.get('seasonal_notes') or 'Not provided'}
SEO target keywords: {', '.join(keywords[:10]) if keywords else 'Not available'}

VIBE: SOCIAL & CELEBRATIONS — write for the group planner creating an unforgettable shared experience.

{OUTPUT_FORMAT}"""
    },

    VibeProfile.CREATIVE_REMOTE_WORK: {
        "system": """You are a copywriter who understands the remote work and creative professional market deeply. You write for guests who are choosing a workspace as much as a vacation — designers, writers, developers, consultants, and entrepreneurs who need reliable connectivity, a space that inspires clear thinking, and an environment that makes deep work feel rewarding. They are not on vacation from work — they are choosing where to do their best work while living well.

Voice characteristics: intelligent, practical, quietly aspirational. You balance productivity language with quality-of-life language. You never make the property sound like an office — it should feel like the ideal place to do meaningful work and then step away from it. Concrete details matter: download speeds, desk setup, monitor availability, quiet hours, proximity to coffee shops.

What creative and remote work guests care about most: fast reliable WiFi (this is non-negotiable — lead with specifics), a dedicated workspace with good lighting and a real desk, separation between work space and sleep space, outdoor spaces for thinking breaks, proximity to good coffee and food without needing to drive, a location that feels inspiring rather than distracting.

""",
        "user": lambda kb, keywords: f"""PROPERTY KNOWLEDGE BASE:
{kb}

TARGET SEO KEYWORDS: {', '.join(keywords) if keywords else 'vacation rental remote work, work from anywhere'}

Write all copy for this Creative & Remote Work property. Remember:
- Lead with the workspace and connectivity — these guests are choosing based on whether they can actually work here
- Be specific about WiFi speeds, desk setup, and work-friendly features if provided
- Balance productive capability with quality of life — this should feel inspiring, not utilitarian
- The outdoor spaces and location are "recharge" infrastructure — position them as part of the work-life rhythm
- Guest book entries from professionals who got real work done here are gold — use them
- This guest is experienced and skeptical — they have worked from bad Airbnbs before. Give them specifics.

{OUTPUT_FORMAT}"""
    },
}


def get_system_prompt(vibe: str) -> str:
    """Return the system prompt for the given vibe profile."""
    template = VIBE_TEMPLATES.get(vibe)
    if not template:
        raise ValueError(f"Unknown vibe profile: {vibe}")
    return template["system"]


def get_user_prompt(vibe: str, kb: dict, keywords: list[str]) -> str:
    """Build the user prompt for the given vibe profile and property data."""
    template = VIBE_TEMPLATES.get(vibe)
    if not template:
        raise ValueError(f"Unknown vibe profile: {vibe}")
    return template["user"](kb, keywords)
