from tqdm import tqdm
import re
from datetime import datetime
import time
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc, AbsDoc
import httpx
from dotenv import load_dotenv
import os
import os, re, json, math
from typing import List, Dict, Any
from openai import OpenAI


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}' started...")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

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

def scopus_ref_via_ip(doi: str, apikey: str, insttoken: str):
    """
    Try to retrieve full-text JSON from Elsevier Article API using institutional IP entitlement.
    Works only if your public IP is recognized as subscribed.
    """
    headers = {
        "X-ELS-APIKey": apikey,
        "X-ELS-Insttoken": insttoken,
        "Accept": "application/json"
    }

    params = {"view": "REF"}  # ensure REF view is requested
    # params = { }  # ensure REF view is requested
    timeout = httpx.Timeout(100.0, connect=60.0)

    with httpx.Client(timeout=timeout, headers=headers) as client:
        url = f"https://api.elsevier.com/content/abstract/doi/{doi}"
        r = client.get(url, params=params)

        # --- Debug information ---
        print(f"HTTP status: {r.status_code}")
        print(f"Entitlement header: {r.headers.get('X-ELS-Entitlement-Status')}")
        if r.status_code != 200:
            print("Response text (first 500 chars):")
            print(r.text[:500])
            r.raise_for_status()

        return r.json()        


def year_from_coverdate(cd):
    # "2004-01-01" -> "2004"
    return cd.split("-")[0]

def build_ref_index(scopus_refs):
    """Return dict { (surname_lower, year): [ref_obj, ...] } in case of duplicates."""
    index = {}
    for ref in scopus_refs:
        try:
            authors = ref.get("author-list", {}).get("author", [])
            if not authors:
                continue
            first_author = authors[0].get("ce:surname") or authors[0].get("preferred-name", {}).get("ce:surname")
            if not first_author:
                continue
            coverdate = ref.get("prism:coverDate")
            if not coverdate:
                continue
            year = year_from_coverdate(coverdate)
            key = (first_author.lower(), year)
            index.setdefault(key, []).append(ref)
        except Exception:
            continue
    return index


citation_pattern = re.compile(
    r"""
    (?P<full>
        # parenthesized style: (Ho and Ermon, 2016) or (Ziebart et al., 2008a; ...)
        \(
            (?P<inside>[^()]+?)
        \)
        |
        # narrative style: Ho and Ermon (2016)
        (?P<author_before>[A-Z][A-Za-z\-']+(?:\s+et\s+al\.)?(?:\s+and\s+[A-Z][A-Za-z\-']+)*)    # authorish
        \s*\(\s*(?P<year_after>\d{4}[a-z]?)\s*\)
    )
    """,
    re.VERBOSE
)


def find_citations_in_text(text):
    """Yield dicts with 'match', 'start', 'end', 'authors_years'."""
    for m in citation_pattern.finditer(text):
        start, end = m.span()
        if m.group("inside"):
            # could be multiple citations inside parentheses, split by ';'
            inside = m.group("inside")
            parts = [p.strip() for p in inside.split(";")]
            ay = []
            for part in parts:
                # e.g. "Ziebart et al., 2008a"
                # split on comma
                if "," in part:
                    author_part, year_part = part.rsplit(",", 1)
                    author_part = author_part.strip()
                    year_part = year_part.strip()
                else:
                    # fallback, skip
                    continue
                # take first surname
                first_surname = author_part.split()[0]
                ay.append((first_surname, year_part))
            yield {
                "text": text[start:end],
                "start": start,
                "end": end,
                "authors_years": ay,
            }
        else:
            # narrative style
            authors_str = m.group("author_before")
            year = m.group("year_after")
            first_surname = authors_str.split()[0]
            yield {
                "text": text[start:end],
                "start": start,
                "end": end,
                "authors_years": [(first_surname, year)],
            }



def map_citations_to_refs(text, scopus_refs):
    ref_index = build_ref_index(scopus_refs)
    result = {}  # (surname_lower, year_no_suffix) -> list of occurrences
    for cit in find_citations_in_text(text):
        for (surname, year_raw) in cit["authors_years"]:
            # year might be "2008a" -> strip letter for lookup
            year = re.match(r"(\d{4})", year_raw).group(1) if re.match(r"(\d{4})", year_raw) else year_raw
            key = (surname.lower(), year)
            if key in ref_index:
                result.setdefault(key, []).append({
                    "span": (cit["start"], cit["end"]),
                    "text": cit["text"],
                })
            else:
                # unmatched citation (maybe not in scopus list)
                result.setdefault(("__unmatched__", f"{surname} {year_raw}"), []).append({
                    "span": (cit["start"], cit["end"]),
                    "text": cit["text"],
                })
    return result


def preprocess_refs(refs):
    ref_info = []
    for ref in refs:
        # if not none
        if ref.get("author-list") is None:
            continue
        authors = ref.get("author-list", {}).get("author", [])
        if not authors:
            continue
        first_author = authors[0].get("ce:surname", "").lower()
        year = ref.get("prism:coverDate", "")[:4]
        title = ref.get("title", "")
        ref_info.append({
            "id": ref.get("dc:identifier", ""),
            "first_author": first_author,
            "year": year,
            "title": title,
            "authors": [a.get("ce:surname", "") for a in authors],
            "venue": ref.get("prism:publicationName", ""),
            "evidence_sentences": []
        })
    return ref_info

import re
def split_sentences(text):
    text = re.sub(r'\s+', ' ', text)
    # crude but effective split for academic text
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z\(])', text)
    return sents

def match_citations_in_sentence(sent, ref_info):
    matches = []
    for r in ref_info:
        pattern = rf'\b{re.escape(r["first_author"].capitalize())}.*?\b{r["year"]}\b'
        if re.search(pattern, sent):
            matches.append(r)
    return matches
def build_citation_contexts(text, ref_info, window=1):
    sents = split_sentences(text)
    results = { (r["first_author"], r["year"]): [] for r in ref_info }
    for i, s in enumerate(sents):
        matched_refs = match_citations_in_sentence(s, ref_info)
        if matched_refs:
            # add neighboring sentences if they continue the idea
            context_sents = sents[max(0,i-window): min(len(sents), i+window+1)]
            context = " ".join(context_sents)
            for r in matched_refs:
                results[(r["first_author"], r["year"])].append(context)
    return results


import json
from openai import OpenAI
from typing import List, Dict, Any, Optional

@timeit
def enrich_references_with_llm(
    ref_list: List[Dict[str, Any]],
    ref_block: str,
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    For each reference dict in ref_list, use the LLM to look inside ref_block (the scraped
    reference section from the article) and fill in missing fields.
    """
    client = OpenAI(api_key=api_key)
    enriched: List[Dict[str, Any]] = []


    schema = {
        "name": "references",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "first_author": {"type": "string"},
                            "year": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "array", "items": {"type": "string"}},
                            # "venue": {"type": "string"},
                            # "volume": {"type": "string"},
                            # "issue": {"type": "string"},
                            # "pages": {"type": "string"},
                            "raw_match": {"type": "string"},
                        },
                        "required": [
                            "id",
                            "first_author",
                            "year",
                            "title",
                            "authors",
                            # "venue",
                            # "volume",
                            # "issue",
                            # "pages",
                            "raw_match",
                        ],
                    },
                }
            },
            "required": ["references"],
        },
        "strict": True,
    }

    # Summarize your incomplete list
    compact_refs = [
        {
            "id": r.get("id", ""),
            "first_author": r.get("first_author", ""),
            "year": r.get("year", ""),
            "title": r.get("title", ""),
            "authors": r.get("authors", []),
        }
        for r in ref_list
    ]

    prompt = f"""
You are given a block of references from an academic paper and a list of partially
filled reference entries extracted automatically. Your task is to match each
item in the list to its corresponding full reference in the text and fill
missing information (venue, pages, etc.).

Rules:
- Return a list of JSON objects (same length as the input list) in order.
- Each object must include all required fields.
- Do not invent information that is not visible in the reference text.
- If uncertain, keep the field empty string "".

Reference list (incomplete, to complete):
{json.dumps(compact_refs, ensure_ascii=False, indent=2)}

Reference block (source text):
{ref_block}
"""
    now = time.time()
    # print(f"1. Querying LLM to enrich references...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scholarly reference assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema
        },
    )
    # print(f"1. Response received. - Time taken: {time.time() - now:.2f} seconds")


    try:
        # print("Parsing LLM response...")
        # content = resp.output[0].content[0].text  # type: ignore[attr-defined]
        # parsed = json.loads(content)
        # enriched_refs = parsed.get("references", [])
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        enriched_refs = parsed.get("references", [])
    except Exception:
        print("Failed to parse LLM response, falling back to original references.")
        enriched_refs = ref_list  # fallback if parsing fails

    # keep original evidence_sentences
    for i, r in enumerate(enriched_refs):
        r["evidence_sentences"] = ref_list[i].get("evidence_sentences", [])

    return enriched_refs

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

@timeit
def fill_evidence_sentences_batch(
    refs: List[Dict[str, Any]],
    article_text: str,
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    For each reference in `refs`, ask the LLM to find the sentence(s) in `article_text`
    that most likely cite / discuss that reference, and fill `evidence_sentences`.
    Returns a new list with the same length.
    """
    client = OpenAI(api_key=api_key)

    # schema for the LLM output
    schema = {
        "name": "citation_evidence",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            # "id": {"type": "string"},
                            "first_author": {"type": "string"},
                            "year": {"type": "string"},
                            "title": {"type": "string"},
                            "evidence_sentences": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "first_author",
                            "year",
                            "title",
                            "evidence_sentences",
                        ],
                    },
                }
            },
            "required": ["references"],
        },
        "strict": True,
    }

    # we only need to send the minimal identifying info to help the model match
    compact_refs = [
        {
            # "id": r.get("id", ""),
            "first_author": r.get("first_author", ""),
            "year": r.get("year", ""),
            "title": r.get("title", ""),
        }
        for r in refs
    ]

    prompt = f"""
You are given the FULL BODY TEXT of an article and a list of references
(already cleaned and matched).

Your task:
- For EACH reference, find the sentence(s) in the article body where that reference
  is cited or described.
- A "sentence" here means a natural language sentence from the article body.
- Often citations look like “... as proposed by Ho and Ermon (2016) ...” or
  “(Ziebart et al., 2008a; Ziebart et al., 2008b)”.
- If you cannot find any clear sentence for a reference, return an empty list [] for that one.
- Keep the output in the SAME ORDER as the input list.
- DO NOT quote from the reference section; we want sentences from the main text.

Return JSON with one key "references", where each item has:
{{ "id": "<same as input>", "evidence_sentences": [ "...", ... ] }}

References to locate (in order):
{json.dumps(compact_refs, ensure_ascii=False, indent=2)}

Article body to search in:
{article_text}
"""

    # resp = client.responses.create(
    #     model=model,
    #     input=prompt,
    #     response_format={"type": "json_schema", "json_schema": schema},
    # )
    now = time.time()
    # print(f"2. Querying LLM to find evidence sentences...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scholarly citation assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema
        },
    )
    # print(f"2. Response received. - Time taken: {time.time() - now:.2f} seconds")

    try:
        # print("Parsing LLM response for evidence sentences..."  )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        evidence_list = parsed.get("references", [])
    except Exception:
        # fallback to empty evidence
        print("Failed to parse LLM response, falling back to empty evidence sentences.")
        evidence_list = [{"id": r.get("id", ""), "evidence_sentences": []} for r in refs]

    # set evidence_id as first_author+year+first3wordsoftitle to match
    for i, e in enumerate(evidence_list):
        e["id"] = f"{e.get('first_author','')}{e.get('year','')}{' '.join(e.get('title','').split()[:3])}"
    # now merge back into your refs, preserving everything else
    evidence_by_id = {e["id"]: e["evidence_sentences"] for e in evidence_list}
    merged: List[Dict[str, Any]] = []
    for r in refs:
        # rid = r.get("id", "")
        rid = f"{r.get('first_author','')}{r.get('year','')}{' '.join(r.get('title','').split()[:3])}"
        new_r = dict(r)  # copy
        new_r["evidence_sentences"] = evidence_by_id.get(rid, r.get("evidence_sentences", []))
        merged.append(new_r)

    return merged



import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

@timeit
def score_inspiration(
    title: str,
    abstract: str,
    refs_with_evidence: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Given the paper title+abstract and a list of references that each have
    `evidence_sentences`, ask the LLM to assign an inspiration score 1-5 and a short rationale.
    """
    client = OpenAI(api_key=api_key)

    # build a compact list for the model
    compact_refs = []
    for i, r in enumerate(refs_with_evidence):
        compact_refs.append({
            "id": r.get("id", str(i)),
            "first_author": r.get("first_author", ""),
            "year": r.get("year", ""),
            "title": r.get("title", ""),
            "evidence_sentences": r.get("evidence_sentences", []),
        })

    schema = {
        "name": "inspiration_scores",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            # "id": {"type": "string"},
                            "first_author": {"type": "string"},
                            "year": {"type": "string"},
                            "title": {"type": "string"},
                            "problem_inspiration": {"type": "integer"},      # 1–5
                            "conceptual_inspiration": {"type": "integer"},
                            "method_inspiration": {"type": "integer"},
                            "data_inspiration": {"type": "integer"},
                            "evaluation_inspiration": {"type": "integer"},
                            # "overall_inspiration": {"type": "integer"},
                            # "rationale": {"type": "string"},
                            "problem_inspiration_rationale": {"type": "string"},
                            "conceptual_inspiration_rationale": {"type": "string"},
                            "method_inspiration_rationale": {"type": "string"},
                            "data_inspiration_rationale": {"type": "string"},
                            "evaluation_inspiration_rationale": {"type": "string"},
                        },
                        "required": [
                        "first_author", "year", "title",
                        "problem_inspiration",
                        "conceptual_inspiration",
                        "method_inspiration",
                        "data_inspiration",
                        "evaluation_inspiration",
                        # "overall_inspiration",
                        # "rationale",
                        "problem_inspiration_rationale",
                        "conceptual_inspiration_rationale",
                        "method_inspiration_rationale",
                        "data_inspiration_rationale",
                        "evaluation_inspiration_rationale",
                    ],
                    },
                }
            },
            "required": ["scores"],
        },
        "strict": True,
    }

    prompt = f"""
Paper title:
{title}

Paper abstract:
{abstract}

You will evaluate how each cited paper inspired this paper along several dimensions:
- PROBLEM: Did it define or strongly shape the problem setting, scenario, or research question?
- CONCEPTS/THEORY: Did it provide core concepts, theoretical framing, or key definitions that this paper builds on?
- METHODS: Did it provide algorithms, architectures, loss functions, or procedures that are reused or extended?
- DATA: Did it introduce datasets or data collection protocols that this paper relies on?
- EVALUATION: Did it define evaluation metrics, baselines, or experimental protocols that are adopted here?

Use a 1–5 scale for each:
5 = central, this paper could not exist in its current form without this contribution
4 = clearly important, strongly shapes this paper
3 = relevant but not central
2 = weak or indirect influence
1 = minimal or unclear influence

Then assign an OVERALL inspiration score (1–5) that balances all dimensions.
Do NOT equate "inspiration" only with methods; conceptual and problem-level influence
can be just as important.

Cited papers to evaluate (in order):
{json.dumps(compact_refs, ensure_ascii=False, indent=2)}

For each, look ONLY at its evidence_sentences to judge how it was used.
If the evidence sentence is generic or unrelated, give 1 and say it seems mismatched.
Return JSON with "scores": [...]
"""

    # methodological inspiration + conceptual influence

    # resp = client.responses.create(
    #     model=model,
    #     input=prompt,
    #     response_format={"type": "json_schema", "json_schema": schema},
    # )
    now = time.time()
    # print(f"3. Querying LLM to score inspiration...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scholarly citation scoring assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
    )
    # print(f"3. Response received. - Time taken: {time.time() - now:.2f} seconds")

    try:
        # print("Parsing LLM response for inspiration scores..."  )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        scores = parsed.get("scores", [])
    except Exception:
        # fallback: everyone gets 1
        print("Failed to parse LLM response, falling back to default scores of 1.")
        scores = [
            {"id": r.get("id", str(i)), "score": 1, "rationale": "Fallback: could not parse."}
            for i, r in enumerate(refs_with_evidence)
        ]

    enriched = []
    score_by_id = {}
    for s in scores:
        rid = f"{s.get('first_author','')}{s.get('year','')}{' '.join(s.get('title','').split()[:3])}"
        # score_by_id[rid] = {"score": s["score"], "rationale": s["rationale"]}
        # compute overall score as average of components
        comp_scores = [
            s.get("problem_inspiration", 1),
            s.get("conceptual_inspiration", 1),
            s.get("method_inspiration", 1),
            s.get("data_inspiration", 1),
            s.get("evaluation_inspiration", 1),
        ]
        overall_score = round(
            0.3 * comp_scores[0] + # problem
            0.3 * comp_scores[1] + # conceptual
            0.2 * comp_scores[2] + # method
            0.1 * comp_scores[3] + # data
            0.1 * comp_scores[4],  # evaluation
            2
        )

        rationale_parts = [
            f"Problem: {s.get('problem_inspiration_rationale','')}",
            f"Conceptual: {s.get('conceptual_inspiration_rationale','')}",
            f"Method: {s.get('method_inspiration_rationale','')}",
            f"Data: {s.get('data_inspiration_rationale','')}",
            f"Evaluation: {s.get('evaluation_inspiration_rationale','')}",
        ]
        full_rationale = rationale_parts

        score_by_id[rid] = {"problem_inspiration": s.get("problem_inspiration", 1),
                            "conceptual_inspiration": s.get("conceptual_inspiration", 1),
                            "method_inspiration": s.get("method_inspiration", 1),
                            "data_inspiration": s.get("data_inspiration", 1),
                            "evaluation_inspiration": s.get("evaluation_inspiration", 1),
                            "overall_inspiration": overall_score,
                            "rationale": full_rationale}
        
    for i, r in enumerate(refs_with_evidence):
        # rid = r.get("id", str(i))
        rid = f"{r.get('first_author','')}{r.get('year','')}{' '.join(r.get('title','').split()[:3])}"
        s = score_by_id.get(rid, {
            "problem_inspiration": 1,
            "conceptual_inspiration": 1,
            "method_inspiration": 1,
            "data_inspiration": 1,
            "evaluation_inspiration": 1,
            "overall_inspiration": 1,
            "rationale": "No score returned."
        })
        new_r = dict(r)
        new_r["problem_inspiration"] = s["problem_inspiration"]
        new_r["conceptual_inspiration"] = s["conceptual_inspiration"]
        new_r["method_inspiration"] = s["method_inspiration"]
        new_r["data_inspiration"] = s["data_inspiration"]
        new_r["evaluation_inspiration"] = s["evaluation_inspiration"]
        new_r["inspiration_score"] = s["overall_inspiration"]
        new_r["inspiration_rationale"] = s["rationale"]
        enriched.append(new_r)
        # s = score_by_id.get(rid, {"overall_inspiration": 1, "rationale": "No score returned."})
        # new_r = dict(r)
        # new_r["inspiration_score"] = s["overall_inspiration"]
        # new_r["inspiration_rationale"] = s["rationale"]
        # enriched.append(new_r)

    return enriched


def extract_inspiring_references(
    doi: str,
    api_key: str,
    inst_token: str,
    openai_key: str,
) -> List[Dict[str, Any]]:
    """
    Given a DOI, retrieve the full text and references, and identify the most inspiring references.
    Returns a list of references with inspiration scores and rationales.
    """

    # Initialize client with your API key
    client = ElsClient(api_key=api_key, inst_token=inst_token)

    fulldoc = FullDoc(doi=doi)
    # uri = fulldoc.uri

    try:
        fulldoc.read(client)
        fulltext = fulldoc.data["originalText"]
        title = fulldoc.data["coredata"]["dc:title"]
        abstract = fulldoc.data["coredata"].get("dc:description", "")
        print(f"Successfully retrieved document: {title}")
    except Exception as e:
        print(f"Error retrieving document: {e}")
        exit(1)

    # print title and authors
    authors = fulldoc.data["coredata"].get("dc:creator", "")
    print(f"Title: {title}")
    print(f"Authors: {authors}")

    references = scopus_ref_via_ip(doi, api_key, inst_token)
    ref_list = references["abstracts-retrieval-response"]["references"]["reference"]

    ref_info = preprocess_refs(ref_list)
    ref_block = _extract_reference_block(fulltext)

    ref_updated = enrich_references_with_llm(ref_info, ref_block, api_key=openai_key)
    ref_evidence_filled = fill_evidence_sentences_batch(ref_updated, fulltext, api_key=openai_key)

    ref_scored = score_inspiration(title, abstract, ref_evidence_filled, api_key=openai_key)

    # Print top 5 inspiring references
    sorted_refs = sorted(ref_scored, key=lambda x: x["inspiration_score"], reverse=True)

    output = {
        "doi": doi,
        "title": title,
        "authors": authors,
        "inspiring_references": sorted_refs
    }

    return output


if __name__ == "__main__":

    load_dotenv("config.env")
    API_KEY = os.getenv("ELSEVIER_KEY")  # Replace with your Elsevier API key
    INST_KEY = os.getenv("ELSEVIER_INSTTOKEN")  # Replace with your Elsevier Institution Token if needed
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key

    # doi = "10.1016/j.trc.2021.103091"   # Non-OA example
    # doi = "10.1016/j.tre.2023.103213" # OA example

    doi_list = [
        "10.1016/j.trpro.2015.07.010", 
        "10.1016/j.trc.2021.103114",
        "10.1016/j.trc.2017.08.005",
        "10.1016/j.trc.2023.104354",
        "10.1016/j.trb.2015.06.011",
        "10.1016/j.trc.2014.05.011",
        "10.1016/j.trc.2022.103668",
        "10.1016/j.trc.2023.104049",
        "10.1016/j.tre.2023.103213",
        "10.1016/j.trc.2021.103091" 
    ]

    # inspiring_refs = extract_inspiring_references(doi, API_KEY, INST_KEY, OPENAI_KEY)
    for doi in doi_list:
        print(f"Processing DOI: {doi}")
        inspiring_refs = extract_inspiring_references(doi, API_KEY, INST_KEY, OPENAI_KEY)
        # print(f"Top inspiring references for DOI {doi}:")

        # dump to JSON file
        out_filename = doi.replace("/", "_") + "_inspiring_refs.json"
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(inspiring_refs, f, ensure_ascii=False, indent=2)
        # print(f"Results saved to {out_filename}\n")