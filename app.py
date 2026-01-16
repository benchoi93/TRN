# app.py
import json
import os
import sqlite3
import glob
import secrets
from datetime import datetime
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
import hmac
import hashlib
import base64
import time
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

# if config.env exists, load it
if os.path.exists("config.env"):
    load_dotenv("config.env")

DB_PATH = os.environ.get("DB_PATH", "data/survey.db")
RESPONSES_DIR = os.environ.get("RESPONSES_DIR", "responses")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

# Display controls
INITIAL_SHOW = 10     # show top 10 initially
MAX_SHOW = 30         # available via "Show more"

# Page order and their sorting score keys with descriptions
PAGES: List[Tuple[str, str, str, str, type]] = [
    ("problem",    "Problem / Concept", "problem_concept_inspiration", 
     "Identify papers that inspired the research problem, conceptual framework, or theoretical foundation of the target paper.", int),
    ("method",     "Method",            "method_inspiration", 
     "Identify papers that inspired the methodology, analytical approach, or technical implementation used in the target paper.", int),
    ("evaluation", "Evaluation",        "data_evaluation_inspiration", 
     "Identify papers that inspired the evaluation strategy, data sources, or validation approach used in the target paper.", int),
    ("overall",    "Overall",           "inspiration_score", 
     "Identify papers that provided overall inspiration across multiple aspects (problem, method, and/or evaluation) of the target paper.", float),
]

app = FastAPI()


@app.get("/health")
def health():
    """Simple health check for Fly or load balancer probes."""
    return JSONResponse(content={"status": "ok"})


# ----------------------------
# DB helpers
# ----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db():
    # Ensure parent directory for DB exists (DB_PATH may be overridden by env)
    db_dir = os.path.dirname(DB_PATH) or "data"
    os.makedirs(db_dir, exist_ok=True)
    conn = db()
    conn.execute("""
  CREATE TABLE IF NOT EXISTS surveys (
    token TEXT PRIMARY KEY,
    json_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    submitted_at TEXT
  )
  """)
    conn.execute("""
  CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(token) REFERENCES surveys(token)
  )
  """)
    conn.execute("""
  CREATE TABLE IF NOT EXISTS drafts (
    token TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(token) REFERENCES surveys(token)
  )
  """)
    conn.commit()
    conn.close()


ensure_db()


def make_admin_token() -> str:
    """Create a signed admin session token (expires in 1 hour)."""
    if not ADMIN_KEY:
        return ""
    exp = int(time.time()) + 3600
    msg = f"admin|{exp}".encode("utf-8")
    sig = hmac.new(ADMIN_KEY.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(msg + b"." + sig).decode("utf-8")


def verify_admin_token(tok: str) -> bool:
    """Verify a signed admin token. Returns True if ADMIN_KEY is empty (disabled)."""
    if not ADMIN_KEY:
        return True
    if not tok:
        return False
    try:
        raw = base64.urlsafe_b64decode(tok.encode("utf-8"))
        msg, sig = raw.split(b".", 1)
        expected = hmac.new(ADMIN_KEY.encode("utf-8"), msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return False
        payload = msg.decode("utf-8")
        prefix, exp = payload.split("|", 1)
        if prefix != "admin":
            return False
        if int(exp) < int(time.time()):
            return False
        return True
    except Exception:
        return False


# ----------------------------
# JSON + formatting helpers
# ----------------------------
def html_escape(x) -> str:
    """Robust escaping for str/list/dict/None."""
    if x is None:
        s = ""
    elif isinstance(x, str):
        s = x
    elif isinstance(x, list):
        s = ", ".join(str(i) for i in x)
    elif isinstance(x, dict):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_authors(auth) -> str:
    """Return a human-readable author list, prioritizing '$' names.

    Handles cases where authors is a list of dicts with keys like '@_fa' and '$',
    a single dict with '$', a plain list of strings, or an existing string.
    """
    if auth is None:
        return ""
    if isinstance(auth, list):
        names = []
        for a in auth:
            if isinstance(a, dict):
                name = a.get("$") or a.get("name") or ""
                if not name:
                    name = str(a)
            else:
                name = str(a)
            if name:
                names.append(name)
        return ", ".join(names)
    if isinstance(auth, dict):
        return auth.get("$") or auth.get("name") or ""
    return str(auth)


def cast_score(val, typ):
    try:
        if val is None:
            return typ()
        return typ(val)
    except Exception:
        return typ()


def load_survey(token: str) -> Dict[str, Any]:
    conn = db()
    row = conn.execute("SELECT * FROM surveys WHERE token=?", (token,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Invalid token")
    return dict(row)


def load_paper_json(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        raise HTTPException(status_code=500, detail=f"Missing JSON file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_candidates_sorted(paper: Dict[str, Any], score_key: str, typ, top_n: int) -> List[Dict[str, Any]]:
    refs = paper.get("inspiring_references", []) or []
    refs_sorted = sorted(refs, key=lambda r: cast_score(r.get(score_key, 0), typ), reverse=True)
    return refs_sorted[:top_n]


def get_draft(token: str) -> Dict[str, Any]:
    conn = db()
    row = conn.execute("SELECT payload FROM drafts WHERE token=?", (token,)).fetchone()
    conn.close()
    if not row:
        return {}
    try:
        return json.loads(row["payload"])
    except Exception:
        return {}


def load_latest_response(token: str) -> Dict[str, Any]:
    """Return the most recent response payload for a token, or empty dict."""
    conn = db()
    row = conn.execute("SELECT payload FROM responses WHERE token=? ORDER BY created_at DESC LIMIT 1",
                       (token,)).fetchone()
    conn.close()
    if not row:
        return {}
    try:
        return json.loads(row["payload"])
    except Exception:
        return {}


def upsert_draft(token: str, draft: Dict[str, Any]):
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = db()
    conn.execute(
        "INSERT INTO drafts(token, payload, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(token) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
        (token, json.dumps(draft, ensure_ascii=False), now),
    )
    conn.commit()
    conn.close()


def mark_submitted(token: str, payload: Dict[str, Any]):
    ts = payload.get("timestamp") or (datetime.utcnow().isoformat(timespec="seconds") + "Z")
    conn = db()
    conn.execute(
        "INSERT INTO responses(token, payload, created_at) VALUES (?, ?, ?)",
        (token, json.dumps(payload, ensure_ascii=False), ts),
    )
    conn.execute("UPDATE surveys SET submitted_at=? WHERE token=?", (ts, token))
    conn.commit()
    conn.close()


# ----------------------------
# Routing utilities
# ----------------------------
def next_page_key(current: str) -> str:
    keys = [k for (k, _, _, _, _) in PAGES]
    idx = keys.index(current)
    return keys[min(idx + 1, len(keys) - 1)]


def prev_page_key(current: str) -> str:
    keys = [k for (k, _, _, _, _) in PAGES]
    idx = keys.index(current)
    return keys[max(idx - 1, 0)]


def page_meta(page_key: str):
    for k, label, score_key, description, typ in PAGES:
        if k == page_key:
            return (label, score_key, description, typ)
    raise HTTPException(status_code=404, detail="Unknown page")


def format_ref_for_list(r: Dict[str, Any]) -> str:
    # Prefer a readable canonical label
    fa = r.get("first_author", "")
    yr = r.get("year", "")
    title = r.get("title", "")
    raw = r.get("raw_match", "")
    if raw:
        return f"{fa} ({yr}) — {title} | {raw}"
    return f"{fa} ({yr}) — {title}"


def build_full_ref_datalist_html(paper: Dict[str, Any], datalist_id: str) -> str:
    refs = paper.get("inspiring_references", []) or []
    opts = []
    for r in refs:
        label = html_escape(format_ref_for_list(r))
        opts.append(f'<option value="{label}"></option>')
    return f'<datalist id="{html_escape(datalist_id)}">{"".join(opts)}</datalist>'


def build_full_ref_details_html(paper: Dict[str, Any], max_items: int = 999) -> str:
    refs = paper.get("inspiring_references", []) or []
    items = []
    for r in refs[:max_items]:
        items.append(f"<li>{html_escape(format_ref_for_list(r))}</li>")
    return f"""
    <details class="mt">
      <summary class="muted">Show full reference list from JSON ({len(refs)} items)</summary>
      <ol class="muted">
        {''.join(items)}
      </ol>
    </details>
    """


def select_rationale(ref: Dict[str, Any], page_key: str) -> str:
    """Pick a rationale string from inspiration_rationale based on page.

    - problem -> line starting with "Problem:" and non-empty content
    - method -> line starting with "Method:" and non-empty content
    - evaluation -> line starting with "Evaluation:" and non-empty content
    - overall -> join all non-empty lines with " | "
    """
    rat = ref.get("inspiration_rationale")
    if isinstance(rat, list):
        if page_key in ("problem", "method", "evaluation"):
            pref_map = {"problem": "Problem:", "method": "Method:", "evaluation": "Evaluation:"}
            pref = pref_map[page_key]
            for line in rat:
                if isinstance(line, str) and line.strip().startswith(pref):
                    content = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    if content:
                        return line
            return ""
        else:
            parts = []
            for line in rat:
                if isinstance(line, str):
                    content = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    if content:
                        parts.append(line)
            return " | ".join(parts)
    if isinstance(rat, str):
        return rat.strip()
    return ""


# ----------------------------
# HTML renderer (single-page-per-question)
# ----------------------------
def render_question_page(
    token: str,
    page_key: str,
    paper: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    draft: Dict[str, Any],
) -> str:
    label, score_key, description, _typ = page_meta(page_key)

    title = html_escape(paper.get("title", ""))
    doi = html_escape(paper.get("doi", ""))
    authors = html_escape(format_authors(paper.get("authors", "")))

    # Pre-fill existing selections from draft
    existing = (draft.get("answers", {}).get(page_key, {}) or {})
    # Normalize selected ids to strings so comparisons with candidate ids work
    selected_ids = set(str(x) for x in (existing.get("selected_ids", []) or []))
    # CSV of selected ids for initial hidden input value
    selected_csv = ",".join(selected_ids)
    other_text = html_escape(existing.get("other_text", ""))
    comment = html_escape(existing.get("comment", ""))

    datalist_id = f"fullrefs_{page_key}"
    full_datalist = build_full_ref_datalist_html(paper, datalist_id)
    full_details = build_full_ref_details_html(paper)

    # helper for candidate label
    def cand_label(r: Dict[str, Any]) -> str:
        rid = str(r.get("id", ""))
        fa = r.get("first_author", "")
        yr = r.get("year", "")
        rt = r.get("title", "")
        return f"{fa} ({yr}) — {rt}"

    # Build cards (top INITIAL_SHOW visible; rest hidden until "Show more")
    cards_html = ""
    for i, r in enumerate(candidates):
        # Use the extractor-provided `id` when available; else fall back to raw_match/doi/index.
        id_val = r.get("id")
        if not id_val:
            id_val = r.get("raw_match") or r.get("doi") or f"idx{i}"
        id_val = str(id_val)
        rid = html_escape(id_val)
        text = html_escape(cand_label(r))

        ev = r.get("evidence_sentences", []) or []
        # tooltip: keep short (first 4 sentences)
        tip_lines = [str(x) for x in ev[:4]]
        # Build reasoning block: prefer inspiration_rationale from JSON; then fall back without scores.
        reasoning = select_rationale(r, page_key) or r.get("reasoning") or r.get("explanation") or ""
        if not reasoning:
                reasoning = "No rationale provided; see evidence below."
        tooltip_text = f"Our reasoning:\n - {reasoning}"
        # if score_val is not None:
        #         tooltip_text += f"\n - Score: {score_val}"
        if tip_lines:
            tooltip_text += "\nEvidence:\n • " + "\n • ".join(tip_lines)
        tooltip = html_escape(tooltip_text)
        # Prefill by matching stored selected_ids to the reference id
        checked = "checked" if id_val in selected_ids else ""
        hidden_cls = "hidden" if i >= INITIAL_SHOW else ""

        cards_html += f"""
        <div class="card {hidden_cls}" data-tooltip="{tooltip}">
          <label class="card-inner">
            <input class="pick" type="checkbox" name="selected_ids" value="{rid}" {checked}/>
            <div class="card-text">{text}
              <div class="muted" style="font-size:0.8em;margin-top:6px;">ID: {html_escape(id_val)}</div>
            </div>
          </label>
        </div>
        """

    # show more button if needed
    show_more_btn = ""
    if len(candidates) > INITIAL_SHOW:
        show_more_btn = f"""
        <button type="button" class="btn secondary" id="showMoreBtn">
          Show more (up to {min(MAX_SHOW, len(candidates))})
        </button>
        """

    # Navigation
    keys = [k for (k, _, _, _, _) in PAGES]
    is_first = (page_key == keys[0])
    is_last = (page_key == keys[-1])

    prev_link = f"/{token}/{prev_page_key(page_key)}" if not is_first else ""
    next_action = f"/{token}/{page_key}/save"

    progress_idx = keys.index(page_key) + 1
    progress = f"Step {progress_idx} of {len(keys)}"

    # On last page, button text is "Submit"
    next_label = "Submit" if is_last else "Next"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Inspiration Survey</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #fafafa; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .header {{ padding: 18px; background: #fff; border: 1px solid #ddd; border-radius: 10px; }}
    .box {{ margin-top: 18px; padding: 18px; background: #fff; border: 1px solid #ddd; border-radius: 10px; }}
    .muted {{ color: #666; font-size: 0.95em; }}
    .progress {{ margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    @media (min-width: 900px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .card {{
      background: #fff;
      border: 1px solid #e3e3e3;
      border-radius: 10px;
      padding: 12px;
      position: relative;
    }}
    .card:hover {{
      border-color: #bdbdbd;
      box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}
    .card-inner {{ display: flex; gap: 10px; align-items: flex-start; cursor: pointer; }}
    .card-text {{ line-height: 1.25; }}
    .hidden {{ display: none; }}
    .tooltip {{
      display: none;
      position: absolute;
      left: 12px;
      right: 12px;
      top: 100%;
      margin-top: 6px;
      background: #1f1f1f;
      color: #fff;
      padding: 10px;
      border-radius: 8px;
      font-size: 0.92em;
      white-space: pre-line;
      z-index: 10;
    }}
    .card:hover .tooltip {{ display: block; }}
    .controls {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
    .btn {{ padding: 10px 14px; border-radius: 8px; border: 1px solid #ccc; cursor: pointer; background: #fff; }}
    .btn.primary {{ background: #111; color: #fff; border-color: #111; }}
    .btn.secondary {{ background: #fff; }}
    .btn.link {{ border: none; background: none; color: #111; text-decoration: underline; padding: 0; }}
    textarea, input[type=text] {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }}
    label {{ font-weight: bold; display:block; margin-top: 10px; margin-bottom: 6px; }}
    .count {{ font-weight: bold; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="controls">
      <div>
        <h1>{html_escape(label)}</h1>
        <div class="muted progress">{html_escape(progress)}</div>
      </div>
      <div class="muted">
        <div><b>Target paper:</b> {title}</div>
        <div><b>DOI:</b> {doi}</div>
        <div><b>Authors:</b> {authors}</div>
      </div>
    </div>
    <p style="margin-top: 12px; line-height: 1.5;">{html_escape(description)}</p>
    <p class="muted" style="margin-top: 8px;">Select up to <span class="count" id="maxCount">3</span> inspiring papers (unordered). Hover a card to preview evidence.</p>
  </div>

  <form method="post" action="{html_escape(next_action)}" id="qForm">
    <div class="box">
      <div class="controls">
        <div class="muted">Showing {INITIAL_SHOW} suggestions. {("Click “Show more” to view additional candidates." if len(candidates) > INITIAL_SHOW else "")}</div>
        <div>
          {show_more_btn}
        </div>
      </div>

      <div class="muted" style="margin-top:8px;">
        Selected: <span class="count" id="selCount">0</span> / 3
      </div>

      <div class="grid" style="margin-top:12px;">
        {cards_html}
      </div>
      <!-- Hidden field synced with checkbox selections to ensure submission -->
      <input type="hidden" name="selected_ids_json" id="selected_ids_json" value="{html_escape(selected_csv)}" />

      <!-- tooltip elements appended inside cards using CSS/JS -->
      <script>
        // Attach tooltips (keeps HTML cleaner)
        document.querySelectorAll(".card").forEach(card => {{
          const tip = document.createElement("div");
          tip.className = "tooltip";
          tip.textContent = card.getAttribute("data-tooltip") || "";
          card.appendChild(tip);
        }});
      </script>

        <label>If “Other / Not listed”, specify (optional)</label>
        <input
        type="text"
        name="other_text"
        value="{other_text}"
        list="{html_escape(datalist_id)}"
        placeholder="Start typing to search full reference list (from JSON)"
        />
        {full_datalist}
        {full_details}

      <label>Optional comment</label>
      <textarea name="comment" rows="3" placeholder="Why these papers? (optional)">{comment}</textarea>
    </div>

    <div class="box">
      <div class="controls">
        <div>
          {"<a class='btn link' href='" + html_escape(prev_link) + "'>Back</a>" if not is_first else ""}
        </div>
        <button class="btn primary" type="submit">{html_escape(next_label)}</button>
      </div>
    </div>
  </form>
</div>

<script>
  const picks = Array.from(document.querySelectorAll("input.pick"));
  const selCountEl = document.getElementById("selCount");
  const hiddenSel = document.getElementById("selected_ids_json");

  // Debug: list all pick inputs and their initial values
  try {{
    console.debug("[DEBUG] picks count:", picks.length);
    picks.forEach((p, idx) => console.debug("[DEBUG] pick["+idx+"] value=", p.value, "checked=", p.checked));
  }} catch(e) {{}}

  function updateCount() {{
    const checked = picks.filter(p => p.checked).map(p => p.value);
    const n = checked.length;
    selCountEl.textContent = String(n);
    // Update hidden field (comma-separated)
    if (hiddenSel) hiddenSel.value = checked.join(",");
    // Debug: mirror to console for client-side tracing
    try {{ console.debug("[DEBUG] updateCount checked:", checked, "hiddenSel:", hiddenSel ? hiddenSel.value : null); }} catch(e) {{}}
    // Enforce max 3: if already 3 selected, disable unchecked boxes
    if (n >= 3) {{
      picks.forEach(p => {{
        if (!p.checked) p.disabled = true;
      }});
    }} else {{
      picks.forEach(p => p.disabled = false);
    }}
  }}

  picks.forEach(p => p.addEventListener("change", updateCount));
  updateCount();

  // Ensure hidden field is up-to-date on submit
  const qForm = document.getElementById("qForm");
  if (qForm) {{
    qForm.addEventListener("submit", function() {{ updateCount(); }});
  }}

  const showMoreBtn = document.getElementById("showMoreBtn");
  if (showMoreBtn) {{
    showMoreBtn.addEventListener("click", () => {{
      document.querySelectorAll(".card.hidden").forEach(el => el.classList.remove("hidden"));
      showMoreBtn.remove();
    }});
  }}
</script>
</body>
</html>
"""

# ----------------------------
# Routes
# ----------------------------


@app.get("/{token}", response_class=HTMLResponse)
def entry(token: str):
    # Start at problem/concept
    # Special-case the literal 'admin' so '/admin' is handled by the admin routes
    # Redirect to '/admin/list' (not '/admin') to avoid re-entering this dynamic
    # route (which would otherwise capture the same single-segment path and loop).
    if token == "admin":
        return RedirectResponse(url="/admin/list", status_code=303)
    try:
        load_survey(token)  # validates token
    except HTTPException as e:
        # Return a clear HTML message instead of a plain/blank error page
        return HTMLResponse(f"<p>Invalid token: {html_escape(str(e.detail))}</p><p><a href='/'>Back</a></p>")
    return RedirectResponse(url=f"/{token}/welcome", status_code=303)


# Admin actions: make_tokens (register papers)
@app.post('/admin/make_tokens', response_class=HTMLResponse)
async def admin_make_tokens(request: Request):
    """Scan data/papers/*.json and register any files as surveys using the
    filename (no extension) as the token. Requires admin auth if ADMIN_KEY is set.
    """
    # Auth: cookie or posted admin_key (legacy)
    if ADMIN_KEY:
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            pass
        else:
            # If not authenticated via cookie, require POSTed admin_key
            try:
                form = await request.form()
            except Exception:
                form = {}
            provided = ''
            if hasattr(form, 'get'):
                provided = form.get('admin_key', '')
            if provided != ADMIN_KEY:
                raise HTTPException(status_code=401, detail='Invalid admin key')

    base_dir = os.path.dirname(DB_PATH) or 'data'
    papers_dir = os.path.join(base_dir, 'papers')
    os.makedirs(papers_dir, exist_ok=True)

    pattern = os.path.join(papers_dir, '*.json')
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        return HTMLResponse(f"<p>No JSON files found under {html_escape(pattern)}</p><p><a href='/admin/list'>Back</a></p>")

    conn = db()
    created = []
    updated = []
    try:
        # Build a quick map of existing json_path -> token (normalized paths)
        rows = conn.execute("SELECT token, json_path FROM surveys").fetchall()
        path_to_token = {}
        for r in rows:
            try:
                pnorm = os.path.normpath(r['json_path'])
            except Exception:
                pnorm = r['json_path']
            # path_to_token[pnorm] = r['token']
            path_to_token[pnorm] = secrets.token_urlsafe(16)  # short but unguessable enough for prototype

        now = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        for jp in json_files:
            jp_norm = os.path.normpath(jp)
            if jp_norm in path_to_token:
                # already registered for this file; update json_path and created_at to now
                token = path_to_token[jp_norm]
                conn.execute(
                    "UPDATE surveys SET json_path=?, created_at=? WHERE token=?",
                    (jp, now, token),
                )
                updated.append((token, jp))
            else:
                # create a new random token
                token = secrets.token_urlsafe(16)
                # ensure uniqueness (very unlikely collision)
                while conn.execute("SELECT 1 FROM surveys WHERE token=?", (token,)).fetchone():
                    token = secrets.token_urlsafe(16)
                conn.execute(
                    "INSERT INTO surveys(token, json_path, created_at, submitted_at) VALUES (?, ?, ?, NULL)",
                    (token, jp, now),
                )
                created.append((token, jp))
        conn.commit()
    finally:
        conn.close()

    # Build an HTML summary listing created vs updated
    parts = [f"<p>Processed {len(json_files)} files from {html_escape(papers_dir)}</p>"]
    if created:
        parts.append("<h3>Created tokens</h3><ul>")
        for t, p in created:
            parts.append(f"<li><code>{html_escape(t)}</code> &rarr; {html_escape(p)}</li>")
        parts.append("</ul>")
    if updated:
        parts.append("<h3>Updated</h3><ul>")
        for t, p in updated:
            parts.append(f"<li><code>{html_escape(t)}</code> updated -> {html_escape(p)}</li>")
        parts.append("</ul>")

    parts.append("<p><a href='/admin/list'>Back to list</a></p>")
    return HTMLResponse('\n'.join(parts))


@app.get("/", response_class=HTMLResponse)
def index():
        """Front page with purpose, prompt overview, and token entry."""
        page_items = "".join(
                f"<li><b>{html_escape(label)}</b>: {html_escape(desc)}</li>"
                for (_k, label, _score_key, desc, _typ) in PAGES
        )

        return HTMLResponse(f"""<!doctype html>
<html>
<head><meta charset='utf-8'/><title>Inspiration Survey</title>
    <style>
        body{{font-family:Arial,sans-serif;margin:28px;background:#fafafa;}}
        .box{{max-width:900px;margin:0 auto;padding:18px;background:#fff;border:1px solid #ddd;border-radius:10px;}}
        .muted{{color:#555;}}
        label{{display:block;margin-top:8px;font-weight:bold;}}
        input[type=text], input[type=password]{{width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;}}
        ul,ol{{padding-left:20px;}}
    </style>
</head>
<body>
<div class='box'>
    <h1>Welcome to the Inspiration Survey</h1>
    <p>This survey asks you to identify which prior papers inspired different aspects of a target paper. You will see four short prompts and can select up to three papers for each.</p>

    <h3>What you will be asked</h3>
    <ul>{page_items}</ul>

    <h3>How to decide where a paper fits</h3>
    <ol>
        <li><b>Problem vs. Method</b>: Does the paper shape the research question/framework (Problem) or provide techniques/algorithms/implementations (Method)?</li>
        <li><b>Evaluation</b>: Use this when the influence is mainly about data, experiments, or validation strategy.</li>
        <li><b>Overall</b>: Use when a paper broadly inspired multiple aspects or is a key anchor reference.</li>
    </ol>

    <h3>Flow</h3>
    <ol>
        <li>Enter your survey token below to load the target paper.</li>
        <li>For each prompt, pick up to 3 inspiring papers (hover to see evidence and rationale). You can also add an “Other” entry and optional comments.</li>
        <li>Review the final page and submit; you can go back anytime before submitting.</li>
    </ol>

    <form method='post' action='/go' style='margin-top:14px;'>
        <label>Survey token</label>
        <input type='text' name='token' placeholder='Enter token here' />
        <div style='margin-top:10px;'><button type='submit'>Start survey</button></div>
    </form>

    <hr style='margin:18px 0' />

    <div class='muted'>
        <h3>Admin</h3>
        <p>If you have admin access you can upload new JSON files or manage surveys.</p>
        <p><a href='/admin/upload/'>Upload JSON</a> &nbsp;|&nbsp; <a href='/admin/list'>Manage surveys</a></p>
        <form method='post' action='/admin/list' style='margin-top:8px'>
            <label>Admin key (if required)</label>
            <input type='password' name='admin_key' placeholder='Admin key' />
            <div style='margin-top:8px'><button type='submit'>Enter admin panel</button></div>
        </form>
    </div>

</div>
</body>
</html>
""")


def render_welcome(token: str, survey: Dict[str, Any], paper: Dict[str, Any]) -> str:
        """Participant-facing welcome page for a specific survey token."""
        page_items = "".join(
                f"<li><b>{html_escape(label)}</b>: {html_escape(desc)}</li>"
                for (_k, label, _score_key, desc, _typ) in PAGES
        )
        title = html_escape(paper.get("title", ""))
        doi = html_escape(paper.get("doi", ""))
        authors = html_escape(format_authors(paper.get("authors", "")))

        return f"""<!doctype html>
<html>
<head><meta charset='utf-8'/><title>Inspiration Survey</title>
    <style>
        body{{font-family:Arial,sans-serif;margin:28px;background:#fafafa;}}
        .box{{max-width:900px;margin:0 auto;padding:18px;background:#fff;border:1px solid #ddd;border-radius:10px;}}
        .muted{{color:#555;}}
        label{{display:block;margin-top:8px;font-weight:bold;}}
        input[type=text]{{width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;}}
        ul,ol{{padding-left:20px;}}
    </style>
</head>
<body>
<div class='box'>
    <h1>Welcome to the Inspiration Survey</h1>
    <p>This survey asks you to identify which prior papers inspired different aspects of the target paper below.</p>

    <div class='muted' style='margin:10px 0;'>
        <div><b>Target paper:</b> {title}</div>
        <div><b>DOI:</b> {doi}</div>
        <div><b>Authors:</b> {authors}</div>
    </div>

    <h3>What you will be asked</h3>
    <ul>{page_items}</ul>

    <h3>How to decide where a paper fits</h3>
    <ol>
        <li><b>Problem vs. Method</b>: Does the paper shape the research question/framework (Problem) or provide techniques/algorithms/implementations (Method)?</li>
        <li><b>Evaluation</b>: Use this when the influence is mainly about data, experiments, or validation strategy.</li>
        <li><b>Overall</b>: Use when a paper broadly inspired multiple aspects or is a key anchor reference.</li>
    </ol>

    <h3>Flow</h3>
    <ol>
        <li>You will see one prompt at a time for this paper.</li>
        <li>For each prompt, pick up to 3 inspiring papers (hover to see evidence and rationale). You can also add an “Other” entry and optional comments.</li>
        <li>You can navigate back anytime before submitting. On the last page, submit your responses.</li>
    </ol>

    <div style='margin-top:14px;'>
        <a href='/{html_escape(token)}/problem' style='display:inline-block;padding:10px 14px;background:#111;color:#fff;text-decoration:none;border-radius:8px;'>Start survey</a>
    </div>

</div>
</body>
</html>
"""


@app.post('/go')
def go(token: str = Form(...)):
    t = token.strip()
    if not t:
        return HTMLResponse("<p>No token provided. Go back and enter a token.</p>")
    # validate token exists
    try:
        load_survey(t)
    except HTTPException as e:
        return HTMLResponse(f"<p>Invalid token: {html_escape(str(e.detail))}</p><p><a href='/'>Back</a></p>")
    return RedirectResponse(url=f"/{t}/problem", status_code=303)


# Admin actions: make_tokens (register papers) - placed before dynamic token route


@app.post('/admin/make_tokens', response_class=HTMLResponse)
async def admin_make_tokens(request: Request):
    """Scan data/papers/*.json and register any files as surveys using the
    filename (no extension) as the token. Requires admin auth if ADMIN_KEY is set.
    """
    # Auth: cookie or posted admin_key (legacy)
    if ADMIN_KEY:
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            pass
        else:
            # If not authenticated via cookie, require POSTed admin_key
            try:
                form = await request.form()
            except Exception:
                form = {}
            provided = ''
            if hasattr(form, 'get'):
                provided = form.get('admin_key', '')
            if provided != ADMIN_KEY:
                raise HTTPException(status_code=401, detail='Invalid admin key')

    base_dir = os.path.dirname(DB_PATH) or 'data'
    papers_dir = os.path.join(base_dir, 'papers')
    os.makedirs(papers_dir, exist_ok=True)

    pattern = os.path.join(papers_dir, '*.json')
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        return HTMLResponse(f"<p>No JSON files found under {html_escape(pattern)}</p><p><a href='/admin/list'>Back</a></p>")

    conn = db()
    try:
        now = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        for jp in json_files:
            token = os.path.splitext(os.path.basename(jp))[0]
            conn.execute(
                "INSERT INTO surveys(token, json_path, created_at, submitted_at) VALUES (?, ?, ?, NULL) "
                "ON CONFLICT(token) DO UPDATE SET json_path=excluded.json_path, created_at=excluded.created_at, submitted_at=NULL",
                (token, jp, now),
            )
        conn.commit()
    finally:
        conn.close()

    return HTMLResponse(f"<p>Processed {len(json_files)} files. Tokens registered/updated from {html_escape(papers_dir)}</p><p><a href='/admin/list'>Back to list</a></p>")


@app.get("/{token}/{page_key}", response_class=HTMLResponse)
async def question_page(token: str, page_key: str, redo: str = "", request: Request = None):
    # Special-case reserved prefixes that should be handled by explicit routes
    # (e.g. /admin/list). If the token is a reserved word, forward to the
    # appropriate handler instead of treating it as a survey token.
    if token == "admin":
        # If the path is the admin list, forward handling to the admin_list
        # handler directly (avoid issuing a redirect which could be routed
        # back to this dynamic route and cause a loop).
        if page_key == "list":
            return await admin_list(request)
        # Unknown admin subpath -> 404
        raise HTTPException(status_code=404, detail="Not found")

    survey = load_survey(token)

    # If the path is the welcome page, render the intro
    if page_key == "welcome":
        paper = load_paper_json(survey["json_path"]) if survey else {}
        return render_welcome(token, survey, paper)

    # If the path is the thanks page, render it directly (helps avoid routing order issues)
    if page_key == "thanks":
        paper = load_paper_json(survey["json_path"]) if survey else {}
        return render_thank_you(token, paper)

    # If already submitted and the user hasn't requested a redo, show a confirmation page
    if survey.get("submitted_at") and not redo:
        submitted_at = survey.get("submitted_at")
        # small confirmation page with options to view submitted or redo (prefill answers)
        return HTMLResponse(f"""<!doctype html>
<html>
<head><meta charset='utf-8'/><title>Already submitted</title>
  <style>body{{font-family:Arial,sans-serif;margin:28px;background:#fafafa}}.box{{max-width:800px;margin:0 auto;padding:18px;background:#fff;border:1px solid #ddd;border-radius:8px}}</style>
</head><body>
<div class='box'>
  <h1>Survey already submitted</h1>
  <p class='muted'>This token was previously submitted on <b>{html_escape(submitted_at)}</b>.</p>
  <p>You can view the submitted response or re-do the survey (your previous answers will be pre-filled for editing).</p>
  <p><a href='/{html_escape(token)}/thanks'>View submitted response</a> &nbsp; | &nbsp; <a href='/{html_escape(token)}/{html_escape("problem")}?redo=1'>Re-do survey (prefill answers)</a></p>
</div>
</body></html>""")

    # If redo requested, prefill draft from latest response
    if redo:
        prev = load_latest_response(token)
        if prev and isinstance(prev, dict):
            answers = prev.get("answers") or {}
            if answers:
                upsert_draft(token, {"answers": answers})
                # Clear the submitted flag so subsequent saves are treated as a new submission
                try:
                    conn = db()
                    conn.execute("UPDATE surveys SET submitted_at=NULL WHERE token=?", (token,))
                    conn.commit()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

    # Normal flow: render the requested question page
    paper = load_paper_json(survey["json_path"])
    draft = get_draft(token)

    label, score_key, description, typ = page_meta(page_key)
    candidates = get_candidates_sorted(paper, score_key, typ, top_n=MAX_SHOW)

    return render_question_page(token, page_key, paper, candidates, draft)


# ----------------------------
# Admin actions: make_tokens (register papers) - place before dynamic question route collisions


@app.post('/admin/make_tokens', response_class=HTMLResponse)
async def admin_make_tokens(request: Request):
    """Scan data/papers/*.json and register any files as surveys using the
    filename (no extension) as the token. Requires admin auth if ADMIN_KEY is set.
    """
    # Auth: cookie or posted admin_key (legacy)
    if ADMIN_KEY:
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            pass
        else:
            # If not authenticated via cookie, require POSTed admin_key
            try:
                form = await request.form()
            except Exception:
                form = {}
            provided = ''
            if hasattr(form, 'get'):
                provided = form.get('admin_key', '')
            if provided != ADMIN_KEY:
                raise HTTPException(status_code=401, detail='Invalid admin key')

    base_dir = os.path.dirname(DB_PATH) or 'data'
    papers_dir = os.path.join(base_dir, 'papers')
    os.makedirs(papers_dir, exist_ok=True)

    pattern = os.path.join(papers_dir, '*.json')
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        return HTMLResponse(f"<p>No JSON files found under {html_escape(pattern)}</p><p><a href='/admin/list'>Back</a></p>")

    conn = db()
    try:
        now = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        for jp in json_files:
            token = os.path.splitext(os.path.basename(jp))[0]
            conn.execute(
                "INSERT INTO surveys(token, json_path, created_at, submitted_at) VALUES (?, ?, ?, NULL) "
                "ON CONFLICT(token) DO UPDATE SET json_path=excluded.json_path, created_at=excluded.created_at, submitted_at=NULL",
                (token, jp, now),
            )
        conn.commit()
    finally:
        conn.close()

    return HTMLResponse(f"<p>Processed {len(json_files)} files. Tokens registered/updated from {html_escape(papers_dir)}</p><p><a href='/admin/list'>Back to list</a></p>")


# Admin upload UI + handlers
# ----------------------------


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    """Render the upload page. If admin cookie is present and valid, do not
    require the admin_key in the form; otherwise show the admin_key input.
    """
    authenticated = False
    if ADMIN_KEY:
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            authenticated = True

    note = "" if not ADMIN_KEY else "Admin key is required to upload files."

    if authenticated:
        # show upload form without admin_key; support multiple files & drag/drop
        return HTMLResponse(f"""<!doctype html>
<html>
<head><meta charset='utf-8'/><title>Admin - Upload JSON</title>
        <style>body{{font-family:Arial,sans-serif;margin:28px;background:#fafafa}}.box{{max-width:900px;margin:0 auto;padding:18px;background:#fff;border:1px solid #ddd;border-radius:8px}}label{{display:block;margin-top:8px;font-weight:bold}}#dropzone{{border:2px dashed #ccc;border-radius:8px;padding:18px;text-align:center;cursor:pointer;background:#fff}}</style>
        <script>
            function setupDrop(){{
                const dz = document.getElementById('dropzone');
                const fileInput = document.getElementById('files');
                dz.addEventListener('click', () => fileInput.click());
                dz.addEventListener('dragover', (e) => {{e.preventDefault(); dz.style.borderColor = '#888'}});
                dz.addEventListener('dragleave', (e) => {{e.preventDefault(); dz.style.borderColor = '#ccc'}});
                dz.addEventListener('drop', (e) => {{
                    e.preventDefault(); dz.style.borderColor = '#ccc';
                    const dt = e.dataTransfer; if (!dt) return;
                    const files = dt.files;
                    fileInput.files = files;
                    document.getElementById('fileList').textContent = Array.from(files).map(f=>f.name).join(', ');
                }});
                fileInput.addEventListener('change', () => {{
                    document.getElementById('fileList').textContent = Array.from(fileInput.files).map(f=>f.name).join(', ');
                }});
            }}
            window.addEventListener('load', setupDrop);
        </script>
</head>
<body>
<div class='box'>
        <h1>Admin — Upload JSON and register survey</h1>
        <p class='muted'>{html_escape(note)}</p>
        <form method='post' action='/admin/upload/' enctype='multipart/form-data'>

        <label>Token (unique identifier for this survey). If uploading multiple files this field is ignored; tokens will be derived from filenames.</label>
        <input type='text' name='token' value='' />

        <label>JSON files to upload (drag & drop or click to browse)</label>
        <div id='dropzone'>Drop JSON files here or click to select</div>
        <div style='margin-top:8px'><input id='files' type='file' name='files' accept='application/json' multiple style='display:none' /></div>
        <div class='muted' id='fileList' style='margin-top:8px'></div>

        <label style='margin-top:8px'><input type='checkbox' name='overwrite'/> Overwrite if file exists</label>

        <div style='margin-top:12px;'><button type='submit'>Upload and register</button></div>
        </form>
</div>
</body>
</html>
""")

    # Not authenticated: show admin_key input (POSTs to /admin/upload)
    return HTMLResponse(f"""<!doctype html>
<!doctype html>
<html>
<head><meta charset='utf-8'/><title>Admin - Upload JSON</title>
        <style>body{{font-family:Arial,sans-serif;margin:28px;background:#fafafa}}.box{{max-width:900px;margin:0 auto;padding:18px;background:#fff;border:1px solid #ddd;border-radius:8px}}label{{display:block;margin-top:8px;font-weight:bold}}#dropzone{{border:2px dashed #ccc;border-radius:8px;padding:18px;text-align:center;cursor:pointer;background:#fff}}</style>
        <script>
            function setupDrop(){{
                const dz = document.getElementById('dropzone');
                const fileInput = document.getElementById('files');
                dz.addEventListener('click', () => fileInput.click());
                dz.addEventListener('dragover', (e) => {{e.preventDefault(); dz.style.borderColor = '#888'}});
                dz.addEventListener('dragleave', (e) => {{e.preventDefault(); dz.style.borderColor = '#ccc'}});
                dz.addEventListener('drop', (e) => {{
                    e.preventDefault(); dz.style.borderColor = '#ccc';
                    const dt = e.dataTransfer; if (!dt) return;
                    const files = dt.files;
                    fileInput.files = files;
                    document.getElementById('fileList').textContent = Array.from(files).map(f=>f.name).join(', ');
                }});
                fileInput.addEventListener('change', () => {{
                    document.getElementById('fileList').textContent = Array.from(fileInput.files).map(f=>f.name).join(', ');
                }});
            }}
            window.addEventListener('load', setupDrop);
        </script>
</head>
<body>
<div class='box'>
        <h1>Admin — Upload JSON and register survey</h1>
        <p class='muted'>{html_escape(note)}</p>
        <form method='post' action='/admin/upload/' enctype='multipart/form-data'>
        <label>Admin key (if required)</label>
        <input type='password' name='admin_key' value='' />

        <label>Token (unique identifier for this survey). If uploading multiple files this field is ignored; tokens will be derived from filenames.</label>
        <input type='text' name='token' value='' />

        <label>JSON files to upload (drag & drop or click to browse)</label>
        <div id='dropzone'>Drop JSON files here or click to select</div>
        <div style='margin-top:8px'><input id='files' type='file' name='files' accept='application/json' multiple style='display:none' /></div>
        <div class='muted' id='fileList' style='margin-top:8px'></div>

        <label style='margin-top:8px'><input type='checkbox' name='overwrite'/> Overwrite if file exists</label>

        <div style='margin-top:12px;'><button type='submit'>Upload and register</button></div>
        </form>
</div>
</body>
</html>
""")


@app.get("/admin/upload/", response_class=HTMLResponse)
def admin_upload_get(request: Request):
    """Alias GET endpoint so users can visit /admin/upload directly in the browser.
    Delegates to `admin_page` which renders the upload UI (handles auth cookie).
    """
    # Forward to the main admin page renderer which already handles auth
    return admin_page(request)


# Ensure the non-slash path redirects to the canonical upload path.
@app.get("/admin/upload", response_class=HTMLResponse)
def admin_upload_redirect():
    return RedirectResponse(url="/admin/upload/", status_code=303)


@app.post("/admin/upload/", response_class=HTMLResponse)
async def admin_upload(request: Request,
                       admin_key: str = Form(default=""),
                       token: str = Form(default=""),
                       overwrite: str = Form(default=""),
                       files: List[UploadFile] = File(...),
                       ):
    # Basic admin key check (supports cookie-based session or posted key)
    if ADMIN_KEY:
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            pass
        else:
            if not admin_key or admin_key != ADMIN_KEY:
                raise HTTPException(status_code=401, detail="Invalid admin key")

    # Determine save directory: put JSONs next to DB under a 'papers' folder
    base_dir = os.path.dirname(DB_PATH) or "data"
    papers_dir = os.path.join(base_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)

    overwrite_flag = bool(overwrite)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    conn = db()
    saved = []
    errors = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        for f in files:
            orig_name = f.filename or "upload.json"
            if not orig_name.lower().endswith('.json'):
                orig_name = orig_name + '.json'
            filename = orig_name
            dest_path = os.path.join(papers_dir, filename)

            if os.path.exists(dest_path) and not overwrite_flag:
                errors.append((filename, 'exists'))
                continue

            content = await f.read()
            try:
                parsed = json.loads(content)
            except Exception as e:
                errors.append((filename, f'invalid json: {e}'))
                continue

            try:
                with open(dest_path, 'wb') as out:
                    out.write(content)
            except Exception as e:
                errors.append((filename, f'save failed: {e}'))
                continue

            # Token: always generate a random, unguessable token for uploads
            tok = secrets.token_urlsafe(16)
            # Ensure uniqueness (very unlikely collision)
            while conn.execute("SELECT 1 FROM surveys WHERE token=?", (tok,)).fetchone():
                tok = secrets.token_urlsafe(16)

            if not tok:
                errors.append((filename, 'empty token'))
                continue

            try:
                conn.execute(
                    "INSERT INTO surveys(token, json_path, created_at, submitted_at) VALUES (?, ?, ?, NULL) "
                    "ON CONFLICT(token) DO UPDATE SET json_path=excluded.json_path, created_at=excluded.created_at, submitted_at=NULL",
                    (tok, dest_path, now),
                )
                saved.append((tok, dest_path))
            except Exception as e:
                errors.append((filename, f'db error: {e}'))

        conn.commit()
    finally:
        conn.close()

    parts = [f"<p>Processed {len(files)} uploaded files.</p>"]
    if saved:
        parts.append('<h3>Saved</h3><ul>')
        for t, p in saved:
            parts.append(f"<li><code>{html_escape(t)}</code> &rarr; {html_escape(p)}</li>")
        parts.append('</ul>')
    if errors:
        parts.append('<h3>Errors</h3><ul>')
        for fn, msg in errors:
            parts.append(f"<li>{html_escape(fn)}: {html_escape(msg)}</li>")
        parts.append('</ul>')

    parts.append("<p><a href='/admin/list'>Back to list</a></p>")
    return HTMLResponse('\n'.join(parts))


@app.post("/{token}/{page_key}/save")
async def save_page(
    request: Request,
    token: str,
    page_key: str,
    selected_ids: List[str] = Form(default=[]),
    other_text: str = Form(default=""),
    comment: str = Form(default=""),
    selected_ids_json: str = Form(default=""),
):
    survey = load_survey(token)

    # Optional single-submit lock (uncomment if desired)
    # if survey.get("submitted_at"):
    #     raise HTTPException(status_code=409, detail="This token has already been submitted.")

    # Capture raw form payload for debugging (helps trace client->server mismatch)
    try:
        form = await request.form()
        # FormData may contain multiple values per key; collect lists when appropriate
        form_dict = {}
        for k in form.keys():
            if hasattr(form, "getlist"):
                vals = form.getlist(k)
                form_dict[k] = vals if len(vals) > 1 else (vals[0] if vals else "")
            else:
                form_dict[k] = form.get(k)
        try:
            print(f"[DEBUG] raw_form token={token} page={page_key} form={json.dumps(form_dict, ensure_ascii=False)}")
        except Exception:
            print(f"[DEBUG] raw_form token={token} page={page_key} form={form_dict}")
    except Exception as e:
        print(f"[DEBUG] raw_form failed to parse: {e}")

    # Enforce max 3 server-side as well. Prefer JS-provided CSV if present.
    if selected_ids_json:
        parsed_selected = [s for s in selected_ids_json.split(",") if s]
    else:
        parsed_selected = [s for s in selected_ids if s and s != "OTHER"]

    # Debug: log received selection data for diagnostics
    try:
        print(
            f"[DEBUG] save_page token={token} page={page_key} parsed_selected={parsed_selected} selected_ids_json={selected_ids_json}")
    except Exception:
        pass

    if len(parsed_selected) > 3:
        parsed_selected = parsed_selected[:3]

    draft = get_draft(token)
    draft.setdefault("answers", {})
    draft["answers"][page_key] = {
        "selected_ids": parsed_selected,
        "other_text": other_text.strip(),
        "comment": comment.strip(),
    }
    upsert_draft(token, draft)

    keys = [k for (k, _, _, _, _) in PAGES]
    if page_key == keys[-1]:
        # Final submission: build final payload from draft + target paper
        paper = load_paper_json(survey["json_path"])

        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "target_paper": {
                "doi": paper.get("doi", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
            },
            "answers": draft.get("answers", {}),
        }
        # Persist final response into a single file per token (overwrite on re-do)
        os.makedirs(RESPONSES_DIR, exist_ok=True)
        out_path = os.path.join(RESPONSES_DIR, f"{token}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        mark_submitted(token, payload)

        # Redirect to a thank-you page
        return RedirectResponse(url=f"/{token}/thanks", status_code=303)

    # Go to next page
    return RedirectResponse(url=f"/{token}/{next_page_key(page_key)}", status_code=303)


def render_thank_you(token: str, paper: Dict[str, Any], payload: Dict[str, Any] = None) -> str:
    """Return a small thank-you HTML page after submission.

    If `payload` is provided, render the submitted answers for inspection.
    """
    title = html_escape(paper.get("title", ""))
    doi = html_escape(paper.get("doi", ""))
    authors = html_escape(format_authors(paper.get("authors", "")))
    answers_html = ""
    if payload:
        ans = payload.get("answers", {})
        rows = []
        for k, label, _, _ in PAGES:
            a = ans.get(k, {}) or {}
            sel = a.get("selected_ids", []) or []
            sel = [html_escape(str(x)) for x in sel]
            ot = html_escape(a.get("other_text", ""))
            cm = html_escape(a.get("comment", ""))
            rows.append(
                f"<tr><td><b>{html_escape(label)}</b></td><td>{', '.join(sel) or '<i>None</i>'}</td><td>{ot or '<i>—</i>'}</td><td>{cm or '<i>—</i>'}</td></tr>")
        answers_html = """
    <h3>Submitted answers</h3>
    <table border="1" cellpadding="6" style="border-collapse:collapse;">
      <tr><th>Question</th><th>Selected IDs</th><th>Other</th><th>Comment</th></tr>
      {rows}
    </table>
    """.replace('{rows}', ''.join(rows))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Thank you</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #fafafa; }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    .box {{ padding: 18px; background: #fff; border: 1px solid #ddd; border-radius: 8px; }}
    .muted {{ color: #666; }}
    a.button {{ display:inline-block; margin-top:12px; padding:8px 12px; background:#0078d4; color:#fff; text-decoration:none; border-radius:6px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="box">
    <h1>Thanks for your response</h1>
    <p class="muted">Your answers have been recorded. Thank you for helping us build better inspiration datasets.</p>
    <div><b>Target paper:</b> {title}</div>
    <div><b>DOI:</b> {doi}</div>
    <div class="muted"><b>Authors:</b> {authors}</div>
    {answers_html}
    <p><a class="button" href="/{token}">Return to start</a> <a class="button" href="/{token}/problem?redo=1" style="margin-left:8px;background:#666;">Re-do (prefill)</a></p>
  </div>
</div>
</body>
</html>
"""


@app.get("/api/draft/{token}")
def api_get_draft(token: str):
    """Return the current draft payload for a token as JSON (read-only debug endpoint).

    Useful for quick verification of what the server will pre-fill on a re-do without
    querying the DB manually.
    """
    # Validate token exists
    load_survey(token)
    draft = get_draft(token)
    return JSONResponse(content=draft)


@app.get("/api/response/{token}")
def api_get_response(token: str):
    """Return the saved final response file for a token (responses/{token}.json) if present."""
    # Validate token exists
    load_survey(token)
    path = os.path.join(RESPONSES_DIR, f"{token}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No response file found for this token")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load response file: {e}")


@app.get("/{token}/thanks", response_class=HTMLResponse)
def thank_you(token: str):
    # Show a compact thank-you page; validates token
    survey = load_survey(token)
    paper = load_paper_json(survey["json_path"]) if survey else {}
    payload = load_latest_response(token)
    return render_thank_you(token, paper, payload)


@app.api_route("/admin/list", methods=["GET", "POST"], response_class=HTMLResponse)
async def admin_list(request: Request):
    """Admin listing. If ADMIN_KEY is set, require login via POST (form) or
    via a valid signed cookie. Successful login sets an HttpOnly cookie and
    redirects to the list without exposing the key in the URL.
    """
    # Use module-level helpers make_admin_token() and verify_admin_token()

    # If no admin key configured, show list openly
    if ADMIN_KEY:
        # Check cookie first
        cookie = request.cookies.get("admin_auth")
        if cookie and verify_admin_token(cookie):
            authenticated = True
        else:
            authenticated = False

        # If the request is POST (login form), validate and set cookie
        if request.method == "POST":
            form = await request.form()
            provided = form.get("admin_key", "")
            if provided == ADMIN_KEY:
                resp = RedirectResponse(url="/admin/list", status_code=303)
                resp.set_cookie("admin_auth", make_admin_token(), httponly=True, max_age=3600)
                return resp
            # fallthrough to render login with error

        # If admin_key present in query params (legacy), accept it but redirect to clean URL
        qp_key = request.query_params.get("admin_key", "")
        if qp_key:
            if qp_key == ADMIN_KEY:
                resp = RedirectResponse(url="/admin/list", status_code=303)
                resp.set_cookie("admin_auth", make_admin_token(), httponly=True, max_age=3600)
                return resp
            # invalid key in URL -> show login form with error

        if not authenticated:
            # show minimal login prompt (POST form)
            return HTMLResponse("""<!doctype html><html><body style='font-family:Arial,sans-serif;margin:28px'>
    <h2>Admin — Enter key</h2>
    <form method='post' action='/admin/list'>
      <input name='admin_key' type='password' placeholder='admin key'/>
      <button type='submit'>Enter</button>
    </form>
    </body></html>""")

    # At this point either ADMIN_KEY is empty or the user is authenticated
    conn = db()
    rows = conn.execute(
        "SELECT token, json_path, created_at, submitted_at FROM surveys ORDER BY created_at DESC").fetchall()
    conn.close()

    items = []
    for r in rows:
        token = html_escape(r['token'])
        path = html_escape(r['json_path'])
        created = html_escape(r['created_at'] or '')
        submitted = html_escape(r['submitted_at'] or '')
        # Delete form uses cookie-based auth now; do not include admin_key in the form
        items.append(
            f"<tr><td>{token}</td><td>{created}</td><td>{submitted}</td><td><a href='/{token}/problem'>Open</a></td><td>{path}</td><td>"
            f"<form method='post' action='/admin/delete' style='display:inline'>"
            f"<input type='hidden' name='token' value='{token}'/>"
            f"<label style='margin-left:8px;'><input type='checkbox' name='delete_files'/> Remove files</label> "
            f"<button type='submit' style='margin-left:8px;'>Delete</button></form></td></tr>"
        )

    table = """
<!doctype html>
<html><body style='font-family:Arial,sans-serif;margin:28px'>
<h1>Surveys</h1>
    <p>
    <form method='get' action='/admin/upload/' style='display:inline;margin-right:8px'>
        <button type='submit' style='padding:6px 10px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer;'>Upload JSON</button>
    </form>
    &nbsp;|
    <form method='post' action='/admin/make_tokens' style='display:inline;margin-left:12px'>
        <button type='submit'>Make tokens from data/papers</button>
    </form>
    &nbsp;|
    <form method='get' action='/admin/clear_db' style='display:inline;margin-left:12px'>
        <button type='submit' style='background:#b22222;color:#fff;padding:6px;border-radius:6px;border:none;'>Clear DB</button>
    </form>
    </p>
<table border='1' cellpadding='6' style='border-collapse:collapse; width:100%'>
<tr><th>Token</th><th>Created</th><th>Submitted</th><th>Open</th><th>JSON path</th><th>Actions</th></tr>
{rows}
</table>
</body></html>
""".replace('{rows}', ''.join(items))

    return HTMLResponse(table)


@app.post('/admin/delete', response_class=HTMLResponse)
def admin_delete(request: Request, admin_key: str = Form(default=''), token: str = Form(default=''), delete_files: str = Form(default='')):
    if ADMIN_KEY:
        # Allow authentication either via posted admin_key (legacy) or via signed cookie
        cookie = request.cookies.get('admin_auth')
        if cookie and verify_admin_token(cookie):
            pass
        else:
            if not admin_key or admin_key != ADMIN_KEY:
                raise HTTPException(status_code=401, detail='Invalid admin key')

        token = token.strip()
        if not token:
            raise HTTPException(status_code=400, detail='Token required')

        conn = db()
        try:
            row = conn.execute('SELECT json_path FROM surveys WHERE token=?', (token,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail='Token not found')
            json_path = row['json_path']

            # Delete DB records
            conn.execute('DELETE FROM surveys WHERE token=?', (token,))
            conn.execute('DELETE FROM responses WHERE token=?', (token,))
            conn.execute('DELETE FROM drafts WHERE token=?', (token,))
            conn.commit()
        finally:
            conn.close()

        # Optionally delete files on disk
        removed = []
        if delete_files:
            try:
                if os.path.exists(json_path):
                    os.remove(json_path)
                    removed.append(json_path)
            except Exception:
                pass
            # remove response file
            resp = os.path.join(RESPONSES_DIR, f"{token}.json")
            try:
                if os.path.exists(resp):
                    os.remove(resp)
                    removed.append(resp)
            except Exception:
                pass

        return HTMLResponse(f"""<!doctype html><html><body style='font-family:Arial,sans-serif;margin:28px'>
    <h2>Deleted</h2>
    <p>Token: <b>{html_escape(token)}</b></p>
    <p>Removed files: <pre>{html_escape(', '.join(removed) or 'None')}</pre></p>
    <p><a href='/admin/list'>Back to list</a></p>
  </body></html>
  """)

    @app.get('/admin/clear_db', response_class=HTMLResponse)
    def admin_clear_db_page(request: Request):
        """Show a confirmation page to clear all surveys/responses/drafts from the DB.
        Optionally allow removing files on disk. Requires admin auth or shows admin_key input.
        """
        authenticated = False
        if ADMIN_KEY:
            cookie = request.cookies.get('admin_auth')
            if cookie and verify_admin_token(cookie):
                authenticated = True

        if not authenticated and ADMIN_KEY:
            # show minimal login prompt (POST form) that posts back to this endpoint
            return HTMLResponse("""<!doctype html><html><body style='font-family:Arial,sans-serif;margin:28px'>
        <h2>Admin — Enter key to clear DB</h2>
        <form method='post' action='/admin/clear_db'>
          <input name='admin_key' type='password' placeholder='admin key'/>
          <button type='submit'>Enter</button>
        </form>
        </body></html>""")

        # Authenticated: show confirmation form
        return HTMLResponse(f"""<!doctype html><html><body style='font-family:Arial,sans-serif;margin:28px'>
        <h2>Clear survey database</h2>
        <p>This will remove all rows from <code>surveys</code>, <code>responses</code>, and <code>drafts</code>.</p>
        <p>Optionally remove JSON files under the papers folder and saved responses under the responses folder.</p>
        <form method='post' action='/admin/clear_db'>
          <label><input type='checkbox' name='remove_files'/> Also remove JSON and response files on disk</label><br/>
          <div style='margin-top:12px'><button type='submit' style='background:#b22222;color:#fff;padding:8px;border-radius:6px;border:none;'>Confirm clear DB</button></div>
        </form>
        <p><a href='/admin/list'>Cancel</a></p>
        </body></html>""")

    @app.post('/admin/clear_db', response_class=HTMLResponse)
    def admin_clear_db_action(request: Request, admin_key: str = Form(default=''), remove_files: str = Form(default='')):
        """Perform clearing of DB rows and optionally delete files. Requires admin auth.
        """
        # Auth (cookie or posted admin_key)
        if ADMIN_KEY:
            cookie = request.cookies.get('admin_auth')
            if cookie and verify_admin_token(cookie):
                pass
            else:
                if not admin_key or admin_key != ADMIN_KEY:
                    raise HTTPException(status_code=401, detail='Invalid admin key')

        # Count existing rows for reporting
        conn = db()
        try:
            cnt_surveys = conn.execute('SELECT COUNT(*) as c FROM surveys').fetchone()['c']
            cnt_responses = conn.execute('SELECT COUNT(*) as c FROM responses').fetchone()['c']
            cnt_drafts = conn.execute('SELECT COUNT(*) as c FROM drafts').fetchone()['c']

            # Delete rows
            conn.execute('DELETE FROM responses')
            conn.execute('DELETE FROM drafts')
            conn.execute('DELETE FROM surveys')
            conn.commit()
        finally:
            conn.close()

        removed_files = []
        if remove_files:
            # Remove JSON files under data/papers
            base_dir = os.path.dirname(DB_PATH) or 'data'
            papers_dir = os.path.join(base_dir, 'papers')
            if os.path.isdir(papers_dir):
                for fn in os.listdir(papers_dir):
                    fp = os.path.join(papers_dir, fn)
                    try:
                        if os.path.isfile(fp):
                            os.remove(fp)
                            removed_files.append(fp)
                    except Exception:
                        pass
            # Remove response files
            try:
                os.makedirs(RESPONSES_DIR, exist_ok=True)
                for fn in os.listdir(RESPONSES_DIR):
                    fp = os.path.join(RESPONSES_DIR, fn)
                    try:
                        if os.path.isfile(fp):
                            os.remove(fp)
                            removed_files.append(fp)
                    except Exception:
                        pass
            except Exception:
                pass

        parts = [
            f"<p>Cleared database: removed {cnt_surveys} surveys, {cnt_responses} responses, {cnt_drafts} drafts.</p>"]
        if removed_files:
            parts.append('<h3>Removed files</h3><ul>')
            for p in removed_files:
                parts.append(f"<li>{html_escape(p)}</li>")
            parts.append('</ul>')
        parts.append("<p><a href='/admin/list'>Back to list</a></p>")

        return HTMLResponse('\n'.join(parts))

    # (moved) admin_make_tokens is registered earlier to avoid route collision with dynamic token route
