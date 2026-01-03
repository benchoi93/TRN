"""
Streamlit front-end that mirrors the FastAPI survey flow.

Features:
- Lists tokens from `data/survey.db` (table `surveys`)
- Loads/saves drafts to `drafts` table
- Per-question navigation
- Re-do prefill from last response
- Final submit writes `responses/{token}.json` and marks submitted in DB
"""

import os
import json
import datetime
import sqlite3
from typing import List, Dict, Any, Tuple

import streamlit as st

# Mirror the server page order and types
PAGES: List[Tuple[str, str]] = [
    ("problem", "Problem / Concept"),
    ("method", "Method"),
    ("evaluation", "Evaluation"),
    ("overall", "Overall"),
]

DB_PATH = "data/survey.db"
RESPONSES_DIR = "responses"
MAX_SELECTIONS = 3


def db() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def list_surveys() -> List[Dict[str, Any]]:
    conn = db()
    rows = conn.execute(
        "SELECT token, json_path, created_at, submitted_at FROM surveys ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_paper_json(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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


def upsert_draft(token: str, draft: Dict[str, Any]):
    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = db()
    conn.execute(
        "INSERT INTO drafts(token, payload, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(token) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
        (token, json.dumps(draft, ensure_ascii=False), now),
    )
    conn.commit()
    conn.close()


def mark_submitted(token: str, payload: Dict[str, Any]):
    ts = payload.get("timestamp") or (datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z")
    conn = db()
    conn.execute(
        "INSERT INTO responses(token, payload, created_at) VALUES (?, ?, ?)",
        (token, json.dumps(payload, ensure_ascii=False), ts),
    )
    conn.execute("UPDATE surveys SET submitted_at=? WHERE token=?", (ts, token))
    conn.commit()
    conn.close()


st.set_page_config(page_title="Inspiration Survey — Streamlit", layout="wide")
st.title("Inspiration Survey — Streamlit demo")

st.markdown(
    "This front-end mirrors the FastAPI flow: it lists tokens from `data/survey.db`, loads/saves drafts in the `drafts` table, supports per-question navigation, re-do prefill, and writes final responses to `responses/{token}.json` (overwrites on re-do)."
)


# --- load surveys / pick token ---------------------------------
surveys = list_surveys()
if not surveys:
    st.warning("No surveys found in data/survey.db (table `surveys`). Create surveys or run make_tokens.py first.")
    st.stop()

token_map = {s['token']: s for s in surveys}
tokens = list(token_map.keys())
sel_token = st.selectbox("Choose a survey token", options=tokens,
                         format_func=lambda t: f"{t} — {os.path.basename(token_map[t]['json_path'])}")
survey_row = token_map[sel_token]

paper = load_paper_json(survey_row['json_path'])
if not paper:
    st.error(f"Failed to load paper JSON: {survey_row['json_path']}")
    st.stop()

st.subheader(paper.get("title", ""))
st.write("**DOI:**", paper.get("doi", ""))
st.write("**Authors:**", json.dumps(paper.get("authors", []), ensure_ascii=False))


# --- helper to build candidate options -------------------------
def build_options_from_paper(paper: Dict[str, Any]):
    refs = paper.get("inspiring_references", []) or []
    opts = []
    for r in refs:
        rid = r.get("id") or r.get("raw_match") or r.get("doi") or r.get("title") or ""
        label = f"{r.get('first_author','')} ({r.get('year','')}) — {r.get('title','')}"
        opts.append((label, str(rid)))
    return opts


options = build_options_from_paper(paper)


# --- Session / navigation state --------------------------------
if 'page_key' not in st.session_state:
    st.session_state.page_key = PAGES[0][0]

page_keys = [k for k, _ in PAGES]

col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("### Navigation")
    st.write(f"Token: **{sel_token}**")
    submitted = bool(survey_row.get('submitted_at'))
    st.write("Submitted:", survey_row.get('submitted_at') or "—")
    if submitted:
        if st.button("Re-do (prefill from last response)"):
            # Prefill draft from latest response in DB
            conn = db()
            row = conn.execute(
                "SELECT payload FROM responses WHERE token=? ORDER BY created_at DESC LIMIT 1", (sel_token,)).fetchone()
            conn.close()
            if row:
                try:
                    prev = json.loads(row['payload'])
                    answers = prev.get('answers', {})
                    if answers:
                        upsert_draft(sel_token, {"answers": answers})
                        # clear submitted flag so app treats as new
                        conn = db()
                        conn.execute("UPDATE surveys SET submitted_at=NULL WHERE token=?", (sel_token,))
                        conn.commit()
                        conn.close()
                        st.experimental_rerun()
                except Exception:
                    st.warning("Failed to prefill draft from last response")
            else:
                st.warning("No previous response found to prefill from")

    st.markdown("---")
    # Page nav buttons
    if st.button("Previous"):
        idx = page_keys.index(st.session_state.page_key)
        st.session_state.page_key = page_keys[max(0, idx - 1)]
    if st.button("Next"):
        idx = page_keys.index(st.session_state.page_key)
        st.session_state.page_key = page_keys[min(len(page_keys) - 1, idx + 1)]

with col_right:
    # Load draft payload and prefill current page
    draft = get_draft(sel_token) or {}
    answers = draft.get('answers', {}) if isinstance(draft, dict) else {}
    current_key = st.session_state.page_key
    label = dict(PAGES)[current_key]
    st.markdown(f"## {label}")

    # Show top candidate cards as multiselect
    labels = [lab for lab, _ in options]
    vals = [val for _, val in options]

    # prefill selected labels by mapping ids in draft to labels
    existing = answers.get(current_key, {}) or {}
    pre_ids = [str(x) for x in existing.get('selected_ids', []) or []]
    pre_labels = [lab for lab, val in options if val in pre_ids]

    chosen = st.multiselect(f"Select up to {MAX_SELECTIONS}", options=labels,
                            default=pre_labels, key=f"ms_{current_key}")
    # Map chosen labels back to ids
    selected_ids = []
    for ch in chosen:
        for lab, val in options:
            if lab == ch:
                selected_ids.append(val)
                break

    if len(selected_ids) > MAX_SELECTIONS:
        st.error(f"Please select at most {MAX_SELECTIONS} items — you selected {len(selected_ids)}")

    other_text = st.text_input("Other (optional)", value=existing.get('other_text', ''), key=f"other_{current_key}")
    comment = st.text_area("Comment (optional)", value=existing.get('comment', ''), key=f"comment_{current_key}")

    # Save draft button
    if st.button("Save draft"):
        # Merge into draft and upsert
        draft.setdefault('answers', {})
        draft['answers'][current_key] = {
            'selected_ids': selected_ids,
            'other_text': other_text.strip(),
            'comment': comment.strip(),
        }
        upsert_draft(sel_token, draft)
        st.success("Draft saved")

    # If on last page, show Submit
    if current_key == page_keys[-1]:
        if st.button("Submit final response"):
            draft = get_draft(sel_token) or {}
            payload = {
                "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "target_paper": {
                    "doi": paper.get("doi", ""),
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", ""),
                },
                "answers": draft.get('answers', {}),
            }
            os.makedirs(RESPONSES_DIR, exist_ok=True)
            out_path = os.path.join(RESPONSES_DIR, f"{sel_token}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                mark_submitted(sel_token, payload)
                st.success(f"Saved final response to {out_path}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save final response: {e}")

    # Debug / preview
    st.markdown("---")
    st.markdown("### Preview")
    draft = get_draft(sel_token) or {}
    st.json({"draft": draft.get('answers', {})})
