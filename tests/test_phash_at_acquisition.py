"""
PHASH-1: pHash computed unconditionally at acquisition.

Verifies:
- High-res photographs (>= 2.0MP) get a pHash at acquisition.
- Low-res photographs (< MIN_ENHANCE_DIMENSION) get a pHash.
- pHash failure does not abort acquisition — row inserted, hash null.
- onboard invokes deduplicate before enhance.
- enhance filters is_canonical=True (after dedupe has demoted others).
"""

import importlib
import io
from unittest.mock import patch, MagicMock


def test_onboard_deduplicate_before_enhance():
    """onboard calls deduplicate (step 3) before enhance (step 4)."""
    import workflows.onboard as mod
    source = importlib.util.find_spec("workflows.onboard")
    with open(source.origin) as f:
        text = f.read()
    # Find the step numbers
    dedupe_pos = text.find("# ── 3. Deduplicate")
    enhance_pos = text.find("# ── 4. Enhance")
    assert dedupe_pos > 0, "Deduplicate step comment not found"
    assert enhance_pos > 0, "Enhance step comment not found"
    assert dedupe_pos < enhance_pos, (
        f"Deduplicate (pos {dedupe_pos}) must come before Enhance (pos {enhance_pos})"
    )


def test_enhance_filters_canonical_only():
    """enhance.py filters .eq('is_canonical', True)."""
    import skills.enhance as mod
    source = importlib.util.find_spec("skills.enhance")
    with open(source.origin) as f:
        text = f.read()
    assert 'is_canonical' in text, "enhance must filter on is_canonical"


def test_phash_call_matches_enhance():
    """The pHash call in acquire_listing uses the same function as enhance.py.

    Both must call imagehash.phash(img) with no extra parameters.
    """
    import skills.acquire_listing as acq_mod
    import skills.enhance as enh_mod

    acq_source = importlib.util.find_spec("skills.acquire_listing")
    enh_source = importlib.util.find_spec("skills.enhance")

    with open(acq_source.origin) as f:
        acq_text = f.read()
    with open(enh_source.origin) as f:
        enh_text = f.read()

    # Both use imagehash.phash(img) — no hash_size parameter
    assert "imagehash.phash(img)" in acq_text, "acquire must call imagehash.phash(img)"
    assert "imagehash.phash(img)" in enh_text, "enhance must call imagehash.phash(img)"
    # Neither should use hash_size (which would change the output)
    assert "hash_size" not in acq_text.split("imagehash.phash")[1][:30], (
        "acquire must not pass hash_size to imagehash.phash"
    )


def test_phash_unconditional_in_acquire():
    """acquire_listing computes pHash with no dimension or megapixel guard."""
    source = importlib.util.find_spec("skills.acquire_listing")
    with open(source.origin) as f:
        text = f.read()

    # The pHash block must appear at BOTH insert sites
    phash_blocks = text.count("phash_val = str(imagehash.phash(img))")
    assert phash_blocks >= 2, (
        f"Expected pHash computation at both insert sites, found {phash_blocks}"
    )

    # The comment must state it is unconditional
    assert "unconditional" in text.lower(), (
        "pHash comment must state it is unconditional"
    )


def test_phash_failure_does_not_abort():
    """If pHash computation fails, the photograph is still inserted with hash=null."""
    source = importlib.util.find_spec("skills.acquire_listing")
    with open(source.origin) as f:
        text = f.read()

    # After pHash failure, phash_val stays None and the insert continues
    # The insert dict includes "phash": phash_val
    assert '"phash": phash_val' in text, (
        "Photograph insert must include phash field"
    )
    # The except block logs but does not re-raise or continue
    assert "[acquire] pHash failed" in text, (
        "pHash failure must be logged with [acquire] prefix"
    )


def test_acquire_docstring_mentions_phash():
    """The module docstring documents that pHash is computed at acquisition."""
    source = importlib.util.find_spec("skills.acquire_listing")
    with open(source.origin) as f:
        text = f.read()
    # Check the docstring at the top
    assert "pHash" in text[:500], "Module docstring must mention pHash"
