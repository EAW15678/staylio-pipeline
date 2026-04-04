def _upload_to_pages(slug: str, bundle: bytes) -> Optional[str]:
    """
    Upload to Cloudflare Pages Direct Upload API.
    Requires multipart form with 'manifest' field and file content.
    """
    import json as _json
    try:
        headers = {"Authorization": f"Bearer {CLOUDFLARE_API_KEY}"}
        
        # Extract HTML from zip bundle
        buf = io.BytesIO(bundle)
        with zipfile.ZipFile(buf, "r") as zf:
            html_content = zf.read("index.html")
        
        manifest = _json.dumps({"/index.html": ""})
        
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{CLOUDFLARE_PAGES_BASE}/{PAGES_PROJECT_NAME}/deployments",
                headers=headers,
                files={
                    "manifest": (None, manifest, "application/json"),
                    "/index.html": ("index.html", html_content, "text/html"),
                },
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("id")
    except Exception as exc:
        logger.error(f"[TS-12] Cloudflare Pages upload failed: {exc}")
        return None
