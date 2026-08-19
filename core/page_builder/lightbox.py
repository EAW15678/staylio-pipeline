"""
Lightbox and gallery modal JS for the substrate-native page builder.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/page_builder.py, vocabulary-corrected.

In the new vocabulary, the section name IS the display label and the
filter key. The old GCV-to-display-name translation table (cat_labels)
is eliminated — identity mapping.
"""

from core.page_builder.helpers import _esc_js


def _build_lightbox_gallery_js(gallery_items: list) -> str:
    """Build the lightbox + gallery modal markup and JS.

    Emits ONE <dialog> for the lightbox (single image, prev/next/close)
    and ONE <dialog> for the gallery modal (category tabs, browseable).
    Both use native <dialog> for focus trap, Escape key, and aria-modal.

    gallery_items: list of dicts with url, alt, section (or category) keys.
    """
    if not gallery_items:
        return ""

    # Build JS array of photo objects — section name is both key and label
    photos_js_entries = []
    for item in gallery_items:
        url = _esc_js(item.get("url") or "")
        alt = _esc_js(item.get("alt") or "")
        sec = _esc_js(item.get("section") or item.get("category") or "Extras")
        photos_js_entries.append(f"{{'url':'{url}','alt':'{alt}','cat':'{sec}'}}")
    photos_js = "[" + ",".join(photos_js_entries) + "]"

    # Build section tab list (preserving order of first appearance)
    seen_sections: list[str] = []
    for item in gallery_items:
        sec = item.get("section") or item.get("category") or "Extras"
        if sec not in seen_sections:
            seen_sections.append(sec)

    # Section name IS the display label — no translation table
    cat_tabs_js_entries = []
    for sec in seen_sections:
        label = _esc_js(sec)
        cat_tabs_js_entries.append(f"{{'key':'{_esc_js(sec)}','label':'{label}'}}")
    cat_tabs_js = "[" + ",".join(cat_tabs_js_entries) + "]"

    return f"""
  <!-- Lightbox modal (single image viewer) -->
  <dialog id="staylio-lightbox" aria-label="Photo viewer"
    style="position:fixed;inset:0;width:100vw;height:100vh;max-width:none;max-height:none;
           margin:0;padding:0;border:0;background:rgba(0,0,0,.92);z-index:9999;">
    <div style="width:100%;height:100%;display:flex;flex-direction:column;">
      <div style="display:flex;justify-content:flex-end;padding:1rem 1.5rem;">
        <button id="lb-close" type="button" aria-label="Close"
          style="background:none;border:none;color:rgba(255,255,255,.7);font-size:1.1rem;
                 cursor:pointer;padding:.5rem 1rem;">Close &#10005;</button>
      </div>
      <div style="flex:1;display:flex;align-items:center;justify-content:center;
                  position:relative;overflow:hidden;padding:0 3.5rem;">
        <button id="lb-prev" type="button" aria-label="Previous photo"
          style="position:absolute;left:.5rem;top:50%;transform:translateY(-50%);
                 background:none;border:none;color:rgba(255,255,255,.6);font-size:2rem;
                 cursor:pointer;padding:.75rem;">&#8592;</button>
        <img id="lb-img" src="" alt="" loading="lazy"
          style="max-width:100%;max-height:calc(100vh - 140px);object-fit:contain;" />
        <button id="lb-next" type="button" aria-label="Next photo"
          style="position:absolute;right:.5rem;top:50%;transform:translateY(-50%);
                 background:none;border:none;color:rgba(255,255,255,.6);font-size:2rem;
                 cursor:pointer;padding:.75rem;">&#8594;</button>
      </div>
      <div style="padding:.75rem 1.5rem;text-align:center;">
        <span id="lb-caption" style="font-size:.85rem;color:rgba(255,255,255,.5);"></span>
        <span id="lb-counter" style="font-size:.75rem;color:rgba(255,255,255,.35);
                                     display:block;margin-top:.25rem;"></span>
      </div>
    </div>
  </dialog>

  <!-- Gallery modal (full browseable gallery with category tabs) -->
  <dialog id="staylio-gallery" aria-label="Photo gallery"
    style="position:fixed;inset:0;width:100vw;height:100vh;max-width:none;max-height:none;
           margin:0;padding:0;border:0;background:rgba(0,0,0,.95);z-index:9998;">
    <div style="width:100%;height:100%;display:flex;flex-direction:column;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:.75rem 1.5rem;border-bottom:1px solid rgba(255,255,255,.1);">
        <div id="gal-tabs" style="display:flex;gap:.25rem;flex-wrap:wrap;"></div>
        <button id="gal-close" type="button" aria-label="Close gallery"
          style="background:none;border:none;color:rgba(255,255,255,.7);font-size:1rem;
                 cursor:pointer;padding:.5rem 1rem;white-space:nowrap;">Close &#10005;</button>
      </div>
      <div style="flex:1;display:flex;align-items:center;justify-content:center;
                  position:relative;overflow:hidden;padding:0 3.5rem;">
        <button id="gal-prev" type="button" aria-label="Previous photo"
          style="position:absolute;left:.5rem;top:50%;transform:translateY(-50%);
                 background:none;border:none;color:rgba(255,255,255,.6);font-size:2rem;
                 cursor:pointer;padding:.75rem;">&#8592;</button>
        <img id="gal-img" src="" alt="" loading="lazy"
          style="max-width:100%;max-height:calc(100vh - 160px);object-fit:contain;" />
        <button id="gal-next" type="button" aria-label="Next photo"
          style="position:absolute;right:.5rem;top:50%;transform:translateY(-50%);
                 background:none;border:none;color:rgba(255,255,255,.6);font-size:2rem;
                 cursor:pointer;padding:.75rem;">&#8594;</button>
      </div>
      <div style="padding:.5rem 1.5rem;text-align:center;">
        <span id="gal-info" style="font-size:.8rem;color:rgba(255,255,255,.5);"></span>
        <span id="gal-counter" style="font-size:.7rem;color:rgba(255,255,255,.35);
                                      display:block;margin-top:.15rem;"></span>
      </div>
    </div>
  </dialog>

  <script>
  (function() {{
    var PHOTOS = {photos_js};
    var CATS = {cat_tabs_js};

    /* ── Lightbox ─────────────────────────────────────────── */
    var lbDialog = document.getElementById('staylio-lightbox');
    var lbImg    = document.getElementById('lb-img');
    var lbCap    = document.getElementById('lb-caption');
    var lbCount  = document.getElementById('lb-counter');
    var lbIdx    = 0;

    function showLb(idx) {{
      idx = ((idx % PHOTOS.length) + PHOTOS.length) % PHOTOS.length;
      lbIdx = idx;
      var p = PHOTOS[idx];
      lbImg.src = p.url;
      lbImg.alt = p.alt;
      lbCap.textContent = p.alt;
      lbCount.textContent = (idx + 1) + ' / ' + PHOTOS.length;
    }}

    window.openLightbox = function(idx) {{
      showLb(idx);
      if (lbDialog && !lbDialog.open) lbDialog.showModal();
      document.body.style.overflow = 'hidden';
    }};

    if (lbDialog) {{
      document.getElementById('lb-close').onclick = function() {{ lbDialog.close(); }};
      document.getElementById('lb-prev').onclick  = function() {{ showLb(lbIdx - 1); }};
      document.getElementById('lb-next').onclick  = function() {{ showLb(lbIdx + 1); }};
      lbDialog.addEventListener('close', function() {{
        document.body.style.overflow = '';
      }});
      lbDialog.addEventListener('click', function(e) {{
        if (e.target === lbDialog) lbDialog.close();
      }});
    }}

    /* ── Gallery modal ────────────────────────────────────── */
    var galDialog  = document.getElementById('staylio-gallery');
    var galImg     = document.getElementById('gal-img');
    var galInfo    = document.getElementById('gal-info');
    var galCounter = document.getElementById('gal-counter');
    var galTabs    = document.getElementById('gal-tabs');
    var galFiltered = PHOTOS.slice();
    var galIdx     = 0;
    var galActiveCat = null;

    function buildTabs() {{
      if (!galTabs) return;
      galTabs.innerHTML = '';
      var allBtn = document.createElement('button');
      allBtn.type = 'button';
      allBtn.textContent = 'All';
      allBtn.style.cssText = 'border:1px solid rgba(255,255,255,.3);border-radius:3px;' +
        'padding:.3rem .75rem;font-size:.8rem;cursor:pointer;background:' +
        (galActiveCat === null ? 'rgba(255,255,255,.2)' : 'transparent') +
        ';color:rgba(255,255,255,.8);';
      allBtn.onclick = function() {{ filterGal(null); }};
      galTabs.appendChild(allBtn);
      for (var i = 0; i < CATS.length; i++) {{
        (function(cat) {{
          var btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = cat.label;
          btn.style.cssText = 'border:1px solid rgba(255,255,255,.3);border-radius:3px;' +
            'padding:.3rem .75rem;font-size:.8rem;cursor:pointer;background:' +
            (galActiveCat === cat.key ? 'rgba(255,255,255,.2)' : 'transparent') +
            ';color:rgba(255,255,255,.8);';
          btn.onclick = function() {{ filterGal(cat.key); }};
          galTabs.appendChild(btn);
        }})(CATS[i]);
      }}
    }}

    function filterGal(cat) {{
      galActiveCat = cat;
      galFiltered = cat === null ? PHOTOS.slice() :
        PHOTOS.filter(function(p) {{ return p.cat === cat; }});
      galIdx = 0;
      buildTabs();
      showGal(0);
    }}

    function showGal(idx) {{
      if (galFiltered.length === 0) return;
      idx = ((idx % galFiltered.length) + galFiltered.length) % galFiltered.length;
      galIdx = idx;
      var p = galFiltered[idx];
      galImg.src = p.url;
      galImg.alt = p.alt;
      var catLabel = '';
      for (var i = 0; i < CATS.length; i++) {{
        if (CATS[i].key === p.cat) {{ catLabel = CATS[i].label; break; }}
      }}
      galInfo.textContent = catLabel + (catLabel ? ' — ' : '') + p.alt;
      galCounter.textContent = (idx + 1) + ' / ' + galFiltered.length;
    }}

    window.openGallery = function() {{
      galActiveCat = null;
      galFiltered = PHOTOS.slice();
      galIdx = 0;
      buildTabs();
      showGal(0);
      if (galDialog && !galDialog.open) galDialog.showModal();
      document.body.style.overflow = 'hidden';
    }};

    window.openGalleryFiltered = function(cats) {{
      galActiveCat = '__multi__';
      galFiltered = PHOTOS.filter(function(p) {{ return cats.indexOf(p.cat) !== -1; }});
      if (galFiltered.length === 0) galFiltered = PHOTOS.slice();
      galIdx = 0;
      buildTabs();
      showGal(0);
      if (galDialog && !galDialog.open) galDialog.showModal();
      document.body.style.overflow = 'hidden';
    }};

    if (galDialog) {{
      document.getElementById('gal-close').onclick = function() {{ galDialog.close(); }};
      document.getElementById('gal-prev').onclick  = function() {{ showGal(galIdx - 1); }};
      document.getElementById('gal-next').onclick  = function() {{ showGal(galIdx + 1); }};
      galDialog.addEventListener('close', function() {{
        document.body.style.overflow = '';
      }});
      galDialog.addEventListener('click', function(e) {{
        if (e.target === galDialog) galDialog.close();
      }});
    }}

    /* ── Keyboard navigation (shared) ─────────────────────── */
    document.addEventListener('keydown', function(e) {{
      if (lbDialog && lbDialog.open) {{
        if (e.key === 'ArrowLeft')  showLb(lbIdx - 1);
        if (e.key === 'ArrowRight') showLb(lbIdx + 1);
      }}
      if (galDialog && galDialog.open) {{
        if (e.key === 'ArrowLeft')  showGal(galIdx - 1);
        if (e.key === 'ArrowRight') showGal(galIdx + 1);
      }}
    }});
  }})();
  </script>"""
