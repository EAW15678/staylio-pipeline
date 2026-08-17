"""
INTAKE-URLS: URL classifier tests for ingest_intake.

Verifies that listing URLs are classified by hostname, not substring.
The old classifier matched "vacationrentals" as a substring, which
caught PMC sites like vacationrentalslbi.com.
"""


def _classify(url: str):
    """Reproduce the classifier logic from ingest_intake.py."""
    from urllib.parse import urlparse
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        hostname = ""
    if "airbnb.com" in hostname:
        return "airbnb"
    elif ("vrbo.com" in hostname or "homeaway.com" in hostname
          or hostname == "vacationrentals.com"
          or hostname.endswith(".vacationrentals.com")):
        return "vrbo"
    return None


def test_vrbo_dot_com():
    assert _classify("https://www.vrbo.com/4886746") == "vrbo"


def test_vrbo_with_path():
    assert _classify("https://www.vrbo.com/4886746?adults=2") == "vrbo"


def test_vacationrentals_dot_com():
    """vacationrentals.com IS a VRBO-family domain."""
    assert _classify("https://www.vacationrentals.com/listing/12345") == "vrbo"


def test_vacationrentalslbi_is_NOT_vrbo():
    """vacationrentalslbi.com is a PMC, NOT VRBO."""
    assert _classify("https://www.vacationrentalslbi.com/listing.3678") is None


def test_airbnb_dot_com():
    assert _classify("https://www.airbnb.com/rooms/12345") == "airbnb"


def test_airbnb_co_uk():
    assert _classify("https://www.airbnb.co.uk/rooms/12345") is None
    # airbnb.co.uk does NOT contain "airbnb.com" — this is correct,
    # the pipeline only supports .com URLs for scraping


def test_homeaway():
    assert _classify("https://www.homeaway.com/vacation-rental/p12345") == "vrbo"


def test_pmc_website():
    """A bare PMC domain is neither airbnb nor vrbo."""
    assert _classify("https://www.intracoastalrentals.com/rentals/vista-azule") is None


def test_generic_url():
    assert _classify("https://example.com/my-property") is None


def test_empty_url():
    assert _classify("") is None


def test_malformed_url():
    assert _classify("not-a-url") is None
