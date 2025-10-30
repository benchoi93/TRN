import time
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc, AbsDoc
import httpx
from dotenv import load_dotenv
import os
import os, re, json, math
from typing import List, Dict, Any
from openai import OpenAI

# --------- tiny helpers ---------
_SPLIT_SENT = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z(])')  # naive sentence splitter
_NUM_REF_LINE = re.compile(r"^\s*\[(\d{1,4})\]\s+(.*)$") # "[12] Foo bar..."
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_NUMERIC_CITE = re.compile(r"\[(\d{1,4}(?:\s*[\-,]\s*\d{1,4})*)\]")
# e.g. (Smith, 2019), (Smith & Doe, 2020; Lee, 2021)
_AUTHOR_YEAR_CITE = re.compile(r"\(([^\)]*\b(19|20)\d{2}[^\)]*?)\)")

def _extract_reference_block(full_text: str) -> str:
    """
    Heuristic: last occurrence of a references header.
    """
    headers = [r"references", r"REFERENCES", r"References", r"Bibliography"]
    idx = -1
    for h in headers:
        m = list(re.finditer(h, full_text))
        if m:
            idx = max(idx, m[-1].start())
    return full_text[idx:].strip() if idx != -1 else ""

def _parse_references_LLM(reference_block: str, full_text: str, model: str = "gpt-5-nano", api_key = None) -> List[Dict[str, str]]:

    """
    Parses a references section into a canonical list using an LLM.
    Returns list of {id, text, year, first_author}.
    """
    assert api_key is not None, "API key must be provided."
    client = OpenAI(api_key=api_key)

    sys_msg = (
        "You are a scholarly assistant that extracts reference entries from a references section. "
        "Given raw text of references, return a JSON array of objects with fields: "
        "'id' (a unique identifier you create), 'text' (the full reference text), "
        "'year' (publication year), and 'authors' (last names of all authors)."
        "'title' (the title of the work), and 'venue' (the journal or conference name)."
    )

    user_msg = {
        "role": "user",
        "content": f"Here is the references section:\n\n{reference_block}\n\n"
                   f"Here is the full text:\n\n{full_text}\n\n"
                   "Return the parsed references as a JSON array."
    }

    schema = {
        "name": "ParsedReferences",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "year": {"type": "string"},
                            "first_author": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "venue": {"type": "string"},
                            "evidence_sentences": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["id", "text", "year", "first_author", "title", "authors", "venue", "evidence_sentences"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["references"],
            "additionalProperties": False
        }
    }
    response_format={
        "type": "json_schema",
        "json_schema": schema
    }
    now = time.time()
    print("Sending request to LLM for reference parsing...")
    resp= client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_msg}, user_msg],
        response_format=response_format
    )
    print(f"Response received. - Time taken: {time.time() - now:.2f} seconds")

    content = resp.choices[0].message.content
    data = json.loads(content)

    return data


def _parse_references(reference_block: str) -> List[Dict[str, str]]:
    """
    Parses a references section into a canonical list.
    Supports:
      - numbered: [1] Foo...
      - author-year: fallback lines (derive a synthetic id like Smith2019)
    Returns list of {id, text, year, first_author}.
    """
    lines = [ln.strip() for ln in reference_block.splitlines() if ln.strip()]
    refs = []
    seen_ids = set()
    for ln in lines:
        m = _NUM_REF_LINE.match(ln)
        if m:
            rid = f"[{m.group(1)}]"
            text = m.group(2).strip()
        else:
            # author-year fallback
            y = _YEAR.search(ln)
            year = y.group(0) if y else ""
            fa_m = re.match(r"^([A-Z][A-Za-z\-']+)", ln)
            fa = fa_m.group(1) if fa_m else "Ref"
            rid = f"{fa}{year}" if year else fa
            text = ln
        if rid in seen_ids:
            # avoid duplicates from wrapped lines
            continue
        seen_ids.add(rid)
        y = _YEAR.search(text)
        refs.append({
            "id": rid,
            "text": text,
            "year": y.group(0) if y else "",
            "first_author": (re.match(r"^([A-Z][A-Za-z\-']+)", text) or [None, ""])[1]
        })
    return refs

def _expand_numeric_group(s: str) -> List[str]:
    # "3,4" or "5-7" → ["[3]","[4]","[5]","[6]","[7]"]
    items, out = re.split(r"\s*,\s*", s), []
    for p in items:
        if re.search(r"-|–|—", p):
            a, b = re.split(r"-|–|—", p)
            a, b = int(a.strip()), int(b.strip())
            for n in range(min(a,b), max(a,b)+1):
                out.append(f"[{n}]")
        else:
            out.append(f"[{int(p.strip())}]")
    return out

def _collect_evidence_sentences(body_text: str) -> List[Dict[str, Any]]:
    """
    Returns sentences that contain citations, with extracted tokens.
    Each item: { "sentence": str, "tokens": ["[3]","Smith2019", ...] }
    """
    sents = _SPLIT_SENT.split(body_text)
    evidence = []
    for s in sents:
        tokens = set()
        # numeric cites
        for m in _NUMERIC_CITE.finditer(s):
            for tok in _expand_numeric_group(m.group(1)):
                tokens.add(tok)
        # author-year cites -> synthesize firstAuthor+year token
        for m in _AUTHOR_YEAR_CITE.finditer(s):
            chunk = m.group(1)
            # split ; for multiple
            for c in [x.strip() for x in chunk.split(";")]:
                y = _YEAR.search(c)
                year = y.group(0) if y else ""
                fa_m = re.match(r"([A-Z][A-Za-z\-']+)", c)
                fa = fa_m.group(1) if fa_m else ""
                if fa and year:
                    tokens.add(f"{fa}{year}")
        if tokens:
            evidence.append({"sentence": s.strip(), "tokens": sorted(tokens)})
    return evidence



def get_top_inspiring_references(title: str, abstract: str, full_text: str, k: int = 5, model: str = "gpt-4o-mini", api_key = None) -> List[Dict[str, Any]]:
    """
    Identify top-K most 'inspiring' references for a paper whose full_text includes a References section.
    Returns a list of {ref_id, title_or_citation, inspiration_score, rationale, evidence_sentences}.
    Requires OPENAI_API_KEY in env.
    """
    assert api_key is not None, "API key must be provided."
    client = OpenAI(api_key=api_key)

    # 1) Split body vs references
    ref_block = _extract_reference_block(full_text)
    body = full_text.replace(ref_block, "") if ref_block else full_text

    # 2) Parse references & evidence sentences
    # refs = _parse_references(ref_block or full_text[-20000:])  # fallback to tail if header missing
    refs = _parse_references_LLM(ref_block,full_text, model=model, api_key=api_key) if ref_block else []
    if not refs:
        raise ValueError("Could not find any reference entries in the provided text.")
    # evidence = _collect_evidence_sentences(body)

    # Trim to keep prompts manageable
    # cap references and evidence if extremely large
    # MAX_REFS = 300
    # MAX_EVID = 1200
    refs_trim = refs["references"]
    evidences_trim = []
    for refs in refs_trim:
        evidences = []
        for ev in refs["evidence_sentences"]:
            if len(evidences) >= 10:
                break
            evidences.append(ev)
        refs["evidence_sentences"] = evidences
        evidences_trim.extend(evidences)
    # evid_trim = evidence[:MAX_EVID]

    # 3) Ask the model to map evidence -> references and score 'inspiration'
    # Enforce structured output with JSON Schema (Structured Outputs).
    # See: https://platform.openai.com/docs/guides/structured-outputs/examples

    schema = {
        "name": "TopRefs",
        "strict": True,                   # enforce exact schema
        "schema": {
            "type": "object",
            "properties": {
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title":   {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "year":    {"type": "integer"},
                            "doi":     {"type": "string"},
                            "inspiration_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "rationale": {"type": "string"},
                            "evidence_sentences": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["title", "authors", "year", "doi", "inspiration_score", "rationale", "evidence_sentences"],
                        "additionalProperties": False   # <- needed on nested object
                    }
                }
            },
            "required": ["references"],
            "additionalProperties": False               # <- needed on top-level object
        }
    }

    response_format={
        "type": "json_schema",
        "json_schema": schema
    }

    sys_msg = (
        "You are a careful scholarly assistant. Given (A) a reference list and (B) a set of "
        "citation-bearing sentences from the paper body, identify which references most clearly "
        "INSPIRED or were CONSTRUCTIVELY USED by the paper (not merely background). "
        "Score inspiration on [0,1]. Prefer concrete, positive methodological or conceptual adoption signals "
        "like 'we build on', 'we extend', 'following X', 'based on Y', 'adopt the architecture of', "
        "'our loss is adapted from', etc. "
        "Downweight neutral surveys, perfunctory cites, and explicit critiques/limitations."
    )

    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Title of the paper:"},
            {"type": "text", "text": title},
            {"type": "text", "text": "Abstract of the paper:"},
            {"type": "text", "text": abstract},
            {"type": "text", "text": "Reference list (as parsed lines):"},
            {"type": "text", "text": json.dumps(refs_trim, ensure_ascii=False)},  # safe cap
            {"type": "text", "text": "Citation evidence sentences (from the body):"},
            {"type": "text", "text": json.dumps(evidences_trim, ensure_ascii=False)[:400_000]},
            {"type": "text", "text": f"Return ONLY the top {k} references by inspiration_score (descending). "
                                      f"If fewer than {k} are clearly inspiring, return fewer."}
        ]
    }

    now = time.time()
    print("Sending request to LLM...")
    resp= client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_msg}, user_msg],
        response_format=response_format
    )
    print("Response received. - Time taken: {:.2f} seconds".format(time.time() - now))

    import pickle
    pickle.dump(resp, open("debug_resp.pkl", "wb"))
    # content = resp.output[0].content[0].text if hasattr(resp, "output") else resp.output_text  # SDK variants
    content = resp.choices[0].message.content
    data = json.loads(content)

    items = data.get("references", [])
    return items


load_dotenv("config.env")
API_KEY = os.getenv("ELSEVIER_KEY")  # Replace with your Elsevier API key

# Initialize client with your API key
client = ElsClient(API_KEY)

# Create a document object with the Scopus EID or DOI
# Example using EID

# doi = "10.1016/j.trc.2021.103091"   # Non-OA example
doi = "10.1016/j.tre.2023.103213" # OA example
fulldoc = FullDoc(doi=doi)
uri = fulldoc.uri
# absdoc = AbsDoc(uri=uri)


# # Read the document data
# if doc.read(client):
#     # Access the reference list
#     references = doc.data['references']
#     for ref in references:
#         # Process each reference as needed
#         print(ref.get('sourcetitle'), ref.get('title'))
# else:
#     print("Failed to read document data.")

# doc.data["coredata"]
# absdoc = absdoc.read(client)

fulldoc.read(client)
fulltext = fulldoc.data["originalText"]
title = fulldoc.data["coredata"]["dc:title"]
abstract = fulldoc.data["coredata"].get("dc:description", "")

openapi_key = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key
response = get_top_inspiring_references(title, abstract, fulltext, k=5, model="gpt-4o-mini", api_key=openapi_key)

for i, ref in enumerate(response):
    print(f"Rank {i+1}:")
    print(f" Title: {ref['title']}")
    print(f" Authors: {', '.join(ref['authors'])} ({ref['year']})")
    print(f" DOI: {ref['doi']}")
    print(f" Inspiration Score: {ref['inspiration_score']:.3f}")
    print(f" Rationale: {ref['rationale']}")
    print(" Evidence Sentences:")
    for sent in ref['evidence_sentences']:
        print(f"  - {sent}")
    print("\n")

