"""
Agent 5 — Website Builder Data Models

LandingPage is the output of Agent 5.
Represents a fully built, deployed Cloudflare Pages property page.

PageBuildInputs collects the outputs of Agents 1, 2, 3, and 4
into a single structure passed to the page builder.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeployMode(str, Enum):
    STAYLIO_SUBDOMAIN = "staylio_subdomain"   # {slug}.staylio.ai (default all tiers)
    CNAME_CUSTOM     = "cname_custom"        # {custom}.pmcdomain.com (Portfolio tier)


class PageStatus(str, Enum):
    PENDING    = "pending"
    BUILDING   = "building"
    DEPLOYED   = "deployed"
    FAILED     = "failed"
    REBUILDING = "rebuilding"   # Re-deployment triggered by content update


@dataclass
class ABTestVariant:
    """A single A/B test variant for GrowthBook."""
    experiment_id: str
    variant_key: str          # "control" | "variant_a" | "variant_b"
    element: str              # Which page element is being tested
    value: str                # The variant content


@dataclass
class CalendarConfig:
    """Configuration for the availability calendar widget."""
    ical_url: Optional[str] = None        # PMC-provided iCal URL
    cache_endpoint: Optional[str] = None  # Cloudflare Worker cache URL
    pms_type: Optional[str] = None        # "guesty" | "hostaway" | "ownerrez" | None
    pms_api_connected: bool = False       # Portfolio tier direct API
    refresh_interval_minutes: int = 30


@dataclass
class PageBuildInputs:
    """
    All inputs needed to build a single property landing page.
    Collected from Agents 1, 2, 3, and 4 outputs.
    """
    # From Agent 1
    knowledge_base: dict = field(default_factory=dict)

    # From Agent 2
    content_package: dict = field(default_factory=dict)

    # From Agent 3
    visual_media_package: dict = field(default_factory=dict)

    # From Agent 4
    local_guide: dict = field(default_factory=dict)

    # Deployment config
    deploy_mode: DeployMode = DeployMode.STAYLIO_SUBDOMAIN
    custom_domain: Optional[str] = None   # Only for CNAME mode


@dataclass
class LandingPage:
    """
    Represents a built and deployed property landing page.
    Stored in Supabase after successful deployment.
    """
    property_id: str
    slug: str
    page_url: str                       # Full public URL
    deploy_mode: DeployMode

    # Content version tracking
    content_version: int = 1
    last_built_at: Optional[str] = None

    # Status
    status: PageStatus = PageStatus.PENDING
    cloudflare_deployment_id: Optional[str] = None
    build_errors: list[str] = field(default_factory=list)

    # A/B test state
    active_experiments: list[ABTestVariant] = field(default_factory=list)

    # Calendar
    calendar_config: Optional[CalendarConfig] = None

    # Schema.org
    schema_generated: bool = False

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "slug": self.slug,
            "page_url": self.page_url,
            "deploy_mode": self.deploy_mode,
            "content_version": self.content_version,
            "last_built_at": self.last_built_at,
            "status": self.status,
            "cloudflare_deployment_id": self.cloudflare_deployment_id,
            "build_errors": self.build_errors,
            "schema_generated": self.schema_generated,
        }
