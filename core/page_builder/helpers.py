"""
Utility functions for the substrate-native page builder.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/page_builder.py. No database access.
"""


def _esc(text) -> str:
    """HTML-escape a value for safe embedding."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _esc_js(text) -> str:
    """Escape a string for safe embedding inside a JS string literal (single-quoted).

    Handles LLM-generated alt text that may contain quotes, backticks,
    newlines, or </script> sequences that would break the script block.
    """
    if text is None:
        return ""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("</script>", "<\\/script>")
        .replace("</Script>", "<\\/Script>")
        .replace("</SCRIPT>", "<\\/SCRIPT>")
    )


def _format_description(description: str) -> str:
    """Split description into paragraphs."""
    if not description:
        return ""
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    return "\n".join(f"<p>{_esc(p)}</p>" for p in paragraphs[:5])


def _get_headline_variants(content_package: dict) -> list:
    """Extract alternative headline variants for GrowthBook Experiment 1.
    In Phase 1 we pass an empty list — variants added when experiments are configured.
    """
    return []


def _calendar_widget_js() -> str:
    """Inline JavaScript for the availability calendar widget."""
    return """
(function() {
  const widget = document.getElementById('calendar-widget');
  if (!widget) return;

  const cacheUrl = widget.dataset.cacheUrl;
  if (!cacheUrl) {
    widget.innerHTML = '<p class="calendar-unavailable">Calendar loading...</p>';
    return;
  }

  const MONTH_NAMES = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December'];
  const MAX_ADVANCE = 12; // months forward from today

  const today       = new Date();
  const originYear  = today.getFullYear();
  const originMonth = today.getMonth(); // 0-indexed, never goes backward

  // Today as a comparable "YYYY-MM-DD" string (local date, no UTC drift)
  const todayYMD = originYear + '-'
    + String(originMonth + 1).padStart(2, '0') + '-'
    + String(today.getDate()).padStart(2, '0');

  // offset = how many months past today's month the LEFT panel shows (0 = current)
  let offset = 0;

  // Stored as raw "YYYY-MM-DD" strings — avoids UTC vs local-midnight drift
  let blockedRanges = [];

  function dayYMD(year, month, day) {
    return year + '-'
      + String(month + 1).padStart(2, '0') + '-'
      + String(day).padStart(2, '0');
  }

  // String comparison is safe for ISO dates: "2026-05-02" >= "2026-05-02" etc.
  function isBlocked(ymd) {
    return blockedRanges.some(r => ymd >= r.start && ymd < r.end);
  }

  function buildMonthHTML(year, month) {
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const firstDay    = new Date(year, month, 1).getDay();
    let html = `<div class="calendar-month">`;
    html += `<h3>${MONTH_NAMES[month]} ${year}</h3>`;
    html += `<div class="calendar-grid">`;
    html += ['Su','Mo','Tu','We','Th','Fr','Sa']
              .map(d => `<div class="cal-header">${d}</div>`).join('');
    for (let i = 0; i < firstDay; i++) html += '<div class="cal-day empty"></div>';
    for (let day = 1; day <= daysInMonth; day++) {
      const ymd = dayYMD(year, month, day);
      const past = ymd < todayYMD;
      const cls  = past ? 'cal-day past'
                        : isBlocked(ymd) ? 'cal-day blocked'
                                         : 'cal-day available';
      html += `<div class="${cls}">${day}</div>`;
    }
    html += '</div></div>';
    return html;
  }

  function render() {
    const leftDate  = new Date(originYear, originMonth + offset, 1);
    const rightDate = new Date(originYear, originMonth + offset + 1, 1);

    const atStart = offset === 0;
    const atLimit = (offset + 1) >= MAX_ADVANCE;

    const prevBtn = document.getElementById('cal-prev-btn');
    if (prevBtn) {
      prevBtn.disabled       = atStart;
      prevBtn.style.opacity  = atStart ? '0.35' : '1';
      prevBtn.style.cursor   = atStart ? 'not-allowed' : 'pointer';
    }
    const nextBtn = document.getElementById('cal-next-btn');
    if (nextBtn) {
      nextBtn.disabled       = atLimit;
      nextBtn.style.opacity  = atLimit ? '0.35' : '1';
      nextBtn.style.cursor   = atLimit ? 'not-allowed' : 'pointer';
    }

    const grid = document.getElementById('cal-month-grid');
    if (grid) {
      grid.innerHTML = buildMonthHTML(leftDate.getFullYear(),  leftDate.getMonth())
                     + buildMonthHTML(rightDate.getFullYear(), rightDate.getMonth());
    }
  }

  fetch(cacheUrl)
    .then(r => r.json())
    .then(data => {
      // Keep as strings — no Date construction, no UTC/local-midnight drift
      blockedRanges = (data.blocked_dates || []).map(b => ({
        start: b.start,
        end:   b.end,
      }));

      // Wire navigation ONCE after data loads
      const prevBtn = document.getElementById('cal-prev-btn');
      if (prevBtn) {
        prevBtn.addEventListener('click', function() {
          if (offset > 0) { offset -= 1; render(); }
        });
      }
      const nextBtn = document.getElementById('cal-next-btn');
      if (nextBtn) {
        nextBtn.addEventListener('click', function() {
          if (offset + 1 < MAX_ADVANCE) { offset += 1; render(); }
        });
      }

      render();
    })
    .catch(() => {
      widget.innerHTML = '<p class="calendar-unavailable">Calendar temporarily unavailable.</p>';
    });
})();
"""
