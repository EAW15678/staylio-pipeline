"""
Page CSS for the substrate-native page builder.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/page_builder.py. Mobile-first, minimal, professional.
"""


def _page_css() -> str:
    """Core CSS for the landing page. Mobile-first, minimal, professional."""
    return """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --font-serif: 'Cormorant Garamond', Georgia, serif;
      --font-sans: 'Inter', system-ui, sans-serif;
      --color-text: #1a1a1a;
      --color-muted: #666;
      --color-bg: #fff;
      --color-accent: #2c3e50;
      --color-cta: #1a1a1a;
      --max-width: 1200px;
      --spacing: clamp(2rem, 5vw, 4rem);
    }
    body { font-family: var(--font-sans); color: var(--color-text); background: var(--color-bg); }
    .container { max-width: var(--max-width); margin: 0 auto; padding: 0 1.5rem; }
    h1, h2 { font-family: var(--font-serif); font-weight: 400; }
    h2 { font-size: clamp(1.8rem, 3vw, 2.8rem); margin-bottom: 1.5rem; }

    /* Hero */
    .hero { position: relative; height: 100svh; min-height: 600px; display: flex;
            align-items: flex-end; overflow: hidden; }
    .hero-media { position: absolute; inset: 0; }
    .hero-media img, .hero-media video { width: 100%; height: 100%; object-fit: cover; }
    .hero-overlay { position: absolute; inset: 0;
                    background: linear-gradient(to top, rgba(0,0,0,.7) 0%, rgba(0,0,0,.1) 60%); }
    #hero-cta-overlay {
      position: absolute;
      inset: 0;
      z-index: 3;
      display: flex;
      align-items: flex-end;
      justify-content: center;
      padding-bottom: 2rem;
      pointer-events: none;
    }
    #hero-cta-btn {
      pointer-events: all;
      display: flex;
      align-items: center;
      gap: 12px;
      background: rgba(255,255,255,0.12);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 50px;
      color: #fff;
      padding: 20px 48px;
      font-size: 1.15rem;
      font-family: inherit;
      cursor: pointer;
      letter-spacing: 0.02em;
      transition: background 0.2s;
    }
    #hero-cta-btn:hover {
      background: rgba(255,255,255,0.22);
    }
    .hero-cta-icon {
      font-size: 1.1rem;
      line-height: 1;
    }
    .hero-cta-text {
      line-height: 1;
    }
    /* #hero-replay-btn removed — replaced by #hero-hear-btn (unmute control) */
    #hero-hear-btn {
      position: absolute;
      top: 1.5rem;
      right: 1.5rem;
      z-index: 4;
      background: rgba(255,255,255,0.12);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 50px;
      color: #fff;
      padding: 10px 20px;
      font-size: 0.9rem;
      font-family: inherit;
      cursor: pointer;
      letter-spacing: 0.02em;
      transition: background 0.2s;
    }
    #hero-hear-btn:hover {
      background: rgba(255,255,255,0.22);
    }
    .hero-content { position: relative; z-index: 1; padding: 2rem 1.5rem 3rem;
                    max-width: var(--max-width); margin: 0 auto; width: 100%; color: #fff; }
    .location-tag { font-size: .85rem; letter-spacing: .15em; text-transform: uppercase;
                    opacity: .8; margin-bottom: .5rem; }
    .hero-headline { font-size: clamp(2.2rem, 5vw, 4rem); line-height: 1.1;
                     margin-bottom: .75rem; }
    .hero-tagline { font-size: clamp(1rem, 2vw, 1.25rem); opacity: .85; margin-bottom: 1.5rem; }
    .hero-specs { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;
                  font-size: .9rem; opacity: .9; }
    .hero-specs span::before { content: "\\00B7"; margin-right: 1rem; }
    .hero-specs span:first-child::before { content: ""; margin: 0; }

    /* CTAs */
    .staylio-cta-btn {
      display: inline-block; padding: .9rem 2rem; font-size: 1rem; font-weight: 500;
      background: #fff; color: #1a1a1a; border: 2px solid #fff; cursor: pointer;
      text-decoration: none; transition: all .2s ease; letter-spacing: .05em;
    }
    .staylio-cta-btn:hover { background: transparent; color: #fff; }
    .cta-primary { font-size: 1.1rem; padding: 1rem 2.5rem; }
    .cta-large { font-size: 1.2rem; padding: 1.1rem 3rem; }
    section:not(.hero) .staylio-cta-btn {
      background: var(--color-cta); color: #fff; border-color: var(--color-cta);
    }
    section:not(.hero) .staylio-cta-btn:hover {
      background: transparent; color: var(--color-cta);
    }

    /* Sections */
    section { padding: var(--spacing) 0; }
    section:nth-child(even) { background: #f8f8f6; }

    /* Description */
    .description-text { max-width: 720px; }
    .description-text p { font-size: 1.1rem; line-height: 1.8; margin-bottom: 1.25rem;
                          color: #333; }

    /* Spotlights */
    .spotlight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                      gap: 1.5rem; }
    .spotlight-card { padding: 1.5rem; border: 1px solid #e8e8e4; }
    .spotlight-card h3 { font-family: var(--font-serif); font-size: 1.4rem;
                         margin-bottom: .25rem; }
    .spotlight-feature { font-size: .85rem; color: var(--color-muted); text-transform: uppercase;
                         letter-spacing: .1em; margin-bottom: .75rem; }

    /* Photo Tour — category modules */
    .cat-modules-list { display: flex; flex-direction: column; gap: 3rem; }
    .cat-module-label { font-family: var(--font-serif); font-size: 1.5rem; font-weight: 400;
                        margin-bottom: .75rem; }
    .cat-module-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 4px; }
    .cat-module-grid--solo { grid-template-columns: 1fr; }
    .cat-module-hero { width: 100%; height: 420px; object-fit: cover; cursor: pointer;
                       display: block; transition: opacity .2s; }
    .cat-module-supporting { display: flex; flex-direction: column; gap: 4px; height: 420px; }
    .cat-module-thumb { width: 100%; flex: 1; min-height: 0; object-fit: cover;
                        cursor: pointer; display: block; transition: opacity .2s; }
    .cat-module-hero:hover, .cat-module-thumb:hover { opacity: .85; }
    .view-all-wrap { margin-top: 2.5rem; text-align: center; }
    .cat-module-more { margin-top: .75rem; text-align: center; }
    .cat-module-more-btn { font-size: .85rem; color: var(--color-muted); text-decoration: underline;
                           cursor: pointer; }

    /* Gallery (full / secondary) */
    .gallery-grid { display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: .5rem; }
    .gallery-thumb { width: 100%; height: 200px; object-fit: cover; cursor: pointer;
                     transition: opacity .2s; }
    .gallery-thumb:hover { opacity: .85; }
    .gallery-section-label { margin: 1rem 0 .25rem; font-size: .75rem; font-weight: 500;
                              text-transform: uppercase; letter-spacing: .1em;
                              color: var(--color-muted); }

    /* Reviews */
    .reviews-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 1.5rem; }
    .review-card { padding: 1.5rem; border-left: 3px solid #e8e8e4; }
    .guest-book-card { border-left-color: var(--color-accent); }
    .badge { display: inline-block; padding: .2rem .6rem; font-size: .75rem;
             background: var(--color-accent); color: #fff; margin-bottom: .75rem;
             letter-spacing: .05em; text-transform: uppercase; }
    .review-card blockquote { font-style: italic; line-height: 1.7;
                              margin-bottom: .75rem; color: #444; }
    .review-card cite { font-size: .85rem; color: var(--color-muted); }

    /* Amenities */
    .amenities-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                      gap: 1.5rem; }
    .amenity-item h3 { font-family: var(--font-serif); font-size: 1.2rem;
                       margin-bottom: .4rem; }
    .amenity-item p { font-size: .95rem; color: #555; line-height: 1.6; }

    /* Calendar */
    .availability { text-align: center; }
    .calendar-subtext { font-size: .95rem; color: var(--color-muted); margin-bottom: 1rem; }
    .calendar-helper { font-size: .85rem; color: var(--color-muted); margin-top: .75rem; }
    #calendar-widget { margin: 1.5rem auto; max-width: 100%; }
    .cal-nav { display: flex; justify-content: space-between; margin-bottom: .75rem; }
    .cal-nav-btn { background: transparent; border: 1px solid #ccc; border-radius: 4px;
                   padding: .35rem .85rem; font-size: .9rem; cursor: pointer;
                   color: var(--color-text, #222); }
    .cal-nav-btn:hover:not(:disabled) { border-color: #888; }
    .cal-month-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;
                      text-align: left; }
    @media (max-width: 640px) { .cal-month-grid { grid-template-columns: 1fr; gap: 1.5rem; } }
    .calendar-month h3 { font-family: var(--font-serif); font-size: 1.2rem;
                         margin-bottom: .75rem; text-align: center; }
    .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
    .cal-header { text-align: center; font-size: .75rem; color: var(--color-muted);
                  padding: .35rem 0; font-weight: 500; }
    .cal-day { text-align: center; padding: .5rem .15rem; font-size: .85rem; border-radius: 3px; }
    .cal-day.available { background: #fff; color: #222;
                         border: 1px solid #e0e0e0; cursor: default; }
    .cal-day.available:hover { border-color: #999; }
    .cal-day.blocked { background: #D9534F; color: #fff;
                       border: 1px solid #D9534F; cursor: not-allowed; pointer-events: none; }
    .cal-day.past { background: transparent; color: #ccc;
                    border: 1px solid transparent; }
    .calendar-legend { font-size: .8rem; color: var(--color-muted); margin-top: .75rem;
                       text-align: center; }
    .legend-available, .legend-blocked { display: inline-block; width: 12px; height: 12px;
                                          margin-right: 4px; vertical-align: middle;
                                          border-radius: 2px; }
    .legend-available { background: #fff; border: 1px solid #e0e0e0; }
    .legend-blocked { background: #D9534F; }
    .calendar-cta { margin-top: 2rem; }
    .staylio-cta-full { display: block; width: 100%; text-align: center; box-sizing: border-box; }

    /* Local Guide */
    .area-intro { font-size: 1.05rem; line-height: 1.8; max-width: 680px;
                  margin-bottom: 2rem; color: #444; }
    .dont-miss h3 { font-family: var(--font-serif); font-size: 1.4rem;
                    margin-bottom: 1rem; }
    .dont-miss-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                      gap: 1rem; margin-bottom: 2.5rem; }
    .dont-miss-item { padding: 1.25rem; background: #fff; border: 1px solid #e8e8e4; }
    .dont-miss-item h3 { font-size: 1rem; font-weight: 600; margin-bottom: .4rem; }
    .recommendations-grid { display: grid;
                             grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                             gap: 1.25rem; }
    .rec-card { border: 1px solid #e8e8e4; overflow: hidden; }
    .rec-card img { width: 100%; height: 160px; object-fit: cover; }
    .rec-info { padding: 1rem; }
    .rec-info h4 { font-size: 1rem; font-weight: 500; margin-bottom: .25rem; }
    .rec-meta { font-size: .8rem; color: var(--color-muted); margin-bottom: .4rem; }

    /* Owner Story */
    .story-text { max-width: 680px; }
    .story-text p { font-size: 1.05rem; line-height: 1.85; color: #444; }

    /* FAQs */
    .faq-list { max-width: 720px; }
    .faq-item { border-bottom: 1px solid #e8e8e4; }
    .faq-item summary { padding: 1.1rem 0; font-size: 1rem; font-weight: 500;
                         cursor: pointer; list-style: none; display: flex;
                         justify-content: space-between; align-items: center; }
    .faq-item summary::after { content: "+"; font-size: 1.2rem; color: var(--color-muted); }
    .faq-item[open] summary::after { content: "\\2212"; }
    .faq-item p { padding: 0 0 1.1rem; color: #555; line-height: 1.7; }

    /* Footer */
    .footer-cta { text-align: center; background: var(--color-accent) !important;
                  color: #fff; }
    .footer-cta h2, .footer-cta p { color: #fff; }
    .footer-cta h2 { color: #fff; margin-bottom: .5rem; }
    .footer-cta p { margin-bottom: 1.5rem; opacity: .85; }
    .footer-cta .staylio-cta-btn { background: #fff !important; color: var(--color-accent) !important;
                                   border-color: #fff !important; }
    .site-footer { padding: 1.5rem 0; background: #111; color: #888; text-align: center;
                   font-size: .85rem; }
    .site-footer a { color: #888; }
    .powered-by { opacity: .6; }

    @media (max-width: 600px) {
      .hero-headline { font-size: 2rem; }
      .gallery-grid { grid-template-columns: repeat(2, 1fr); }
      .cat-module-grid, .cat-module-grid--solo { grid-template-columns: 1fr; }
      .cat-module-hero { height: 260px; }
      .cat-module-supporting { flex-direction: row; height: auto; }
      .cat-module-thumb { height: 160px; flex: 1; }
    }
  """
