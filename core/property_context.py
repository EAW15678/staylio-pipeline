"""
Property context resolver for Staylio public endpoints.

Resolves canonical property_id and account_id from either:
  - an explicit property_id (internal/pipeline use)
  - a subdomain (public property page use)

Rules:
  - never trust caller-supplied account_id
  - always derive account_id from the canonical properties table
  - only resolve properties with status in ('draft', 'onboarding', 'active')
  - raise HTTPException with clean 404 if not found
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from core.supabase_store import get_supabase


@dataclass
class PropertyContext:
    property_id: str
    account_id: str
    name: str
    subdomain: Optional[str]


def resolve_property_context(
    property_id: Optional[str] = None,
    subdomain: Optional[str] = None,
) -> PropertyContext:
    """
    Resolve canonical property context for use in public endpoints.

    Accepts either property_id or subdomain — not both required.
    property_id takes precedence if both are supplied.

    Raises HTTPException 400 if neither is provided.
    Raises HTTPException 404 if no matching active property is found.
    """
    if not property_id and not subdomain:
        raise HTTPException(
            status_code=400,
            detail="Either property_id or subdomain is required.",
        )

    query = (
        get_supabase()
        .table("properties")
        .select("id, account_id, name, subdomain, status")
        .in_("status", ["draft", "onboarding", "active"])
    )

    if property_id:
        query = query.eq("id", property_id)
    else:
        query = query.eq("subdomain", subdomain)

    result = query.limit(1).execute()

    if not result.data:
        identifier = property_id or subdomain
        raise HTTPException(
            status_code=404,
            detail=f"Property not found: '{identifier}'.",
        )

    row = result.data[0]
    return PropertyContext(
        property_id=row["id"],
        account_id=row["account_id"],
        name=row["name"],
        subdomain=row.get("subdomain"),
    )
