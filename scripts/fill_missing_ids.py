import os
import json
import re
import hashlib

PAPERS_DIR = os.path.join("data", "papers")
DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)")
ARXIV_RE = re.compile(r"arXiv[:\s]*([0-9]{4}\.\d{4,5}|[a-zA-Z\-]+/\d{7,8})", re.I)


def compute_id_for_ref(ref, idx=0):
    raw_match = ref.get("raw_match", "") or ref.get("title", "") or json.dumps(ref, ensure_ascii=False)
    m = DOI_RE.search(raw_match)
    if m:
        return m.group(1)
    m2 = ARXIV_RE.search(raw_match)
    if m2:
        return f"arXiv:{m2.group(1)}"
    dcid = ref.get("id") or ref.get("dc:identifier") or ""
    if dcid:
        return dcid
    # fallback deterministic hash
    h = hashlib.sha1(raw_match.encode("utf-8")).hexdigest()[:12]
    return f"hash:{h}"


def main():
    if not os.path.exists(PAPERS_DIR):
        print("No data/papers directory found. Nothing to do.")
        return
    files = [f for f in os.listdir(PAPERS_DIR) if f.endswith("_inspiring_refs.json")]
    for fname in files:
        path = os.path.join(PAPERS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
        changed = False
        refs = data.get("inspiring_references", [])
        for i, r in enumerate(refs):
            if not r.get("id"):
                new_id = compute_id_for_ref(r, i)
                r["id"] = new_id
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Updated ids in {path}")
        else:
            print(f"No changes for {path}")


if __name__ == "__main__":
    main()
