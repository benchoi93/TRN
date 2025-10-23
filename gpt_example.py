import re
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from pydantic import BaseModel, Field, ValidationError

# ====== LLM (swap to your provider if needed) ======
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# ============== Data structures ==============

@dataclass
class Reference:
    """A single reference entry with a unique label (how it appears in-text)"""
    label: str                 # e.g., "[12]" or "(Smith et al., 2019)" or "Smith2019"
    authors: Optional[str]     # as available from the reference list
    year: Optional[str]
    title: Optional[str]
    raw: str                   # original reference line
    idx: int = field(default=-1)  # position in the reference list


@dataclass
class CitationMention:
    """One concrete in-text mention of a reference"""
    ref_label: str
    start: int
    end: int
    context_left: str
    mention: str
    context_right: str
    matched_ref_key: Optional[str] = None  # key into ref dict


class CitationJudgment(BaseModel):
    """LLM output schema for one citation mention"""
    role_label: str = Field(
        description="One of: 'inspiration/positive-use', 'extension/builds-upon', 'neutral/background', 'contrast/limitation/critique', 'perfunctory/unclear'"
    )
    inspiration_score: float = Field(
        ge=0.0, le=1.0,
        description="How much this text suggests positive inspiration or constructive use (0.0=not at all, 1.0=strongly inspired/used)."
    )
    limitation_score: float = Field(
        ge=0.0, le=1.0,
        description="How much this text suggests critique/limitation (0.0=none, 1.0=strong critique)."
    )
    rationale: str = Field(
        description="One-short-paragraph justification grounded *only* in the provided context."
    )
    evidence_quote: str = Field(
        description="Verbatim snippet (<= 30 words) from the context that supports the judgment."
    )


# ============== Utility: simple parsing helpers ==============

def parse_references(reference_block: str) -> Dict[str, Reference]:
    """
    Parse a reference list block.
    Supports two common styles:
      (A) numbered: [1] Foo..., [2] Bar...
      (B) author-year freeform; we derive a label like 'Smith2019'
    Returns dict keyed by a normalized label that we can match to in-text forms.
    """
    refs: Dict[str, Reference] = {}

    lines = [ln.strip() for ln in reference_block.splitlines() if ln.strip()]
    numbered_pat = re.compile(r"^\[?(\d{1,4})\]?\s+(.*)$")
    year_pat = re.compile(r"\b(19|20)\d{2}\b")

    for i, raw in enumerate(lines):
        m = numbered_pat.match(raw)
        if m:
            num = m.group(1)
            rest = m.group(2)
            year = None
            ym = year_pat.search(rest)
            if ym:
                year = ym.group(0)
            # naive author extraction: substring before first period or '('
            author_guess = re.split(r"[.(]", rest, maxsplit=1)[0][:120]
            title_guess = rest
            label = f"[{num}]"
            refs[label] = Reference(label=label, authors=author_guess, year=year, title=title_guess, raw=raw, idx=i)
            continue

        # author-year style: make a synthetic key like Smith2019 (first author + year)
        # this is heuristic and can be improved
        # try to locate first "Lastname, F." pattern
        auth_match = re.match(r"^([A-Z][a-zA-Z\-']+)[, ]", raw)
        year = None
        ym = year_pat.search(raw)
        if ym:
            year = ym.group(0)
        first_author = auth_match.group(1) if auth_match else "Ref"
        key = f"{first_author}{year or ''}"
        refs[key] = Reference(label=key, authors=first_author, year=year, title=raw, raw=raw, idx=i)

    return refs


def find_intext_citations(full_text: str) -> List[CitationMention]:
    """
    Extract in-text citations in common styles:
      - numeric: [12], [3,4], [5–7]
      - author-year: (Smith, 2019), (Smith & Doe, 2019; Lee, 2020)
    We return one CitationMention *per referenced item* (expanding grouped cites).
    """
    mentions: List[CitationMention] = []
    # Windows around a match for context
    window = 250

    # Numeric style
    for m in re.finditer(r"\[(\d{1,4}(?:\s*[\-,]\s*\d{1,4})*)\]", full_text):
        start, end = m.span()
        inside = m.group(1)
        # expand "3,4" or "5-7"
        items = []
        parts = re.split(r"\s*,\s*", inside)
        for p in parts:
            if re.search(r"-|–|—", p):
                a, b = re.split(r"-|–|—", p)
                a, b = int(a.strip()), int(b.strip())
                rng = range(min(a, b), max(a, b) + 1)
                items.extend([f"[{i}]" for i in rng])
            else:
                items.append(f"[{int(p.strip())}]")

        for it in items:
            context_left = full_text[max(0, start - window):start]
            context_right = full_text[end:min(len(full_text), end + window)]
            mentions.append(CitationMention(ref_label=it,
                                           start=start, end=end,
                                           context_left=context_left,
                                           mention=full_text[start:end],
                                           context_right=context_right))

    # Author–year style (very approximate)
    # e.g., (Smith, 2019), (Smith & Lee, 2020; Kim, 2021)
    for m in re.finditer(r"\(([^\)]*?\b(19|20)\d{2}[^\)]*?)\)", full_text):
        start, end = m.span()
        inside = m.group(1)
        # split by ';' to handle multiple citations inside one set of parens
        chunks = [c.strip() for c in inside.split(";")]
        for c in chunks:
            # heuristic key: first surname + year
            year_m = re.search(r"\b(19|20)\d{2}\b", c)
            year = year_m.group(0) if year_m else ""
            first_author_m = re.match(r"([A-Z][a-zA-Z\-']+)", c)
            first_author = first_author_m.group(1) if first_author_m else "Ref"
            label = f"{first_author}{year}"
            context_left = full_text[max(0, start - window):start]
            context_right = full_text[end:min(len(full_text), end + window)]
            mentions.append(CitationMention(ref_label=label,
                                           start=start, end=end,
                                           context_left=context_left,
                                           mention=full_text[start:end],
                                           context_right=context_right))

    return mentions


def link_mentions_to_refs(mentions: List[CitationMention], refs: Dict[str, Reference]) -> None:
    """
    Best-effort string-key linking:
      - direct key match
      - for numeric, exact "[n]" keys
      - for author-year synthetic keys, exact match as created in parse_references
    """
    for cit in mentions:
        if cit.ref_label in refs:
            cit.matched_ref_key = cit.ref_label
        else:
            # fallback: try loosy matching numeric only
            if cit.ref_label.startswith("[") and cit.ref_label.endswith("]"):
                # sometimes reference list uses "1." instead of "[1]"
                num = cit.ref_label.strip("[]")
                alt = f"[{num}]"
                if alt in refs:
                    cit.matched_ref_key = alt
            else:
                # try prefix match (e.g., Smith2019 vs Smith2019a)
                candidates = [k for k in refs if k.startswith(cit.ref_label)]
                cit.matched_ref_key = candidates[0] if candidates else None


# ============== LLM Scoring ==============

SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful scholarly assistant that classifies the semantic role of citations. "
         "Given the local context around an in-text citation, decide how the cited work is used. "
         "Be conservative and rely ONLY on the text provided."),
        ("human",
         "Paper context BEFORE citation:\n---\n{left}\n---\n\n"
         "Citation token:\n---\n{mention}\n---\n\n"
         "Paper context AFTER citation:\n---\n{right}\n---\n\n"
         "Task:\nReturn a strict JSON object with fields:\n"
         "{json_schema}\n\n"
         "Guidance:\n"
         "- role_label ∈ {{'inspiration/positive-use','extension/builds-upon','neutral/background','contrast/limitation/critique','perfunctory/unclear'}}\n"
         "- inspiration_score ∈ [0,1] should reflect how much this text suggests positive inspiration or constructive use.\n"
         "- limitation_score ∈ [0,1] should reflect critique/negative use.\n"
         "- evidence_quote should be verbatim and ≤ 30 words.\n"
         "Return ONLY the JSON."
         )
    ]
)

def build_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    # Swap to Anthropic, Google, etc. by changing this constructor.
    return ChatOpenAI(model=model_name, temperature=temperature)


def score_mentions_with_llm(mentions: List[CitationMention], llm) -> Dict[int, CitationJudgment]:
    """
    For each mention, call the LLM and parse a JSON response to CitationJudgment.
    Returns: dict mapping mention_index -> CitationJudgment
    """
    judgments: Dict[int, CitationJudgment] = {}
    json_schema = json.dumps(CitationJudgment.model_json_schema(), indent=2)

    chain = SCORING_PROMPT | llm | StrOutputParser()

    for i, cit in enumerate(mentions):
        try:
            resp = chain.invoke({
                "left": cit.context_left[-1200:],   # keep prompt within limits
                "mention": cit.mention,
                "right": cit.context_right[:1200],
                "json_schema": json_schema
            })
            # try to find the JSON block
            # strip text and load json
            text = resp.strip()
            # Some models may wrap code fences; clean them
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            data = json.loads(text)
            judgments[i] = CitationJudgment(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # fallback: mark as uncertain
            judgments[i] = CitationJudgment(
                role_label="perfunctory/unclear",
                inspiration_score=0.0,
                limitation_score=0.0,
                rationale=f"Parser/validation error: {str(e)}",
                evidence_quote=""
            )
    return judgments


# ============== Aggregation & Ranking ==============

def aggregate_scores(
    mentions: List[CitationMention],
    judgments: Dict[int, CitationJudgment],
    refs: Dict[str, Reference],
    agg: str = "weighted_mean"
) -> pd.DataFrame:
    """
    Aggregate per reference: combine all mention scores into a single 'inspiration_score'
    and 'limitation_score', count of roles, etc. Return a DataFrame ranked by inspiration.
    """
    per_ref: Dict[str, Dict] = defaultdict(lambda: {
        "ref_key": "",
        "ref_title": "",
        "ref_authors": "",
        "ref_year": "",
        "n_mentions": 0,
        "roles": Counter(),
        "insp_scores": [],
        "lim_scores": [],
    })

    for i, cit in enumerate(mentions):
        ref_key = cit.matched_ref_key or cit.ref_label
        j = judgments.get(i)
        if not j:
            continue
        bucket = per_ref[ref_key]
        bucket["ref_key"] = ref_key
        ref = refs.get(ref_key)
        if ref:
            bucket["ref_title"] = ref.title or ""
            bucket["ref_authors"] = ref.authors or ""
            bucket["ref_year"] = ref.year or ""
        bucket["n_mentions"] += 1
        bucket["roles"][j.role_label] += 1
        bucket["insp_scores"].append(j.inspiration_score)
        bucket["lim_scores"].append(j.limitation_score)

    rows = []
    for k, v in per_ref.items():
        if len(v["insp_scores"]) == 0:
            continue
        if agg == "weighted_mean":
            # simple weighting: more mentions → slightly higher weight via sqrt(n)
            n = len(v["insp_scores"])
            insp = float(np.mean(v["insp_scores"])) * (1.0 + 0.1*np.log1p(n))
            lim = float(np.mean(v["lim_scores"])) * (1.0 + 0.1*np.log1p(n))
        else:
            insp = float(np.mean(v["insp_scores"]))
            lim = float(np.mean(v["lim_scores"]))

        rows.append({
            "ref_key": k,
            "ref_title": v["ref_title"],
            "ref_authors": v["ref_authors"],
            "ref_year": v["ref_year"],
            "n_mentions": v["n_mentions"],
            "role_counts": dict(v["roles"]),
            "inspiration_score": round(insp, 4),
            "limitation_score": round(lim, 4),
            "net_inspiration": round(insp - lim, 4),
        })

    df = pd.DataFrame(rows).sort_values(
        by=["inspiration_score", "net_inspiration", "n_mentions"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df


# ============== Orchestrator ==============

def run_pipeline(full_text: str, reference_block: str,
                 model_name: str = "gpt-4o-mini",
                 out_prefix: str = "citation_semantics"):
    refs = parse_references(reference_block)
    mentions = find_intext_citations(full_text)
    link_mentions_to_refs(mentions, refs)

    llm = build_llm(model_name=model_name, temperature=0.0)
    judgments = score_mentions_with_llm(mentions, llm)

    # Aggregate & rank
    df_rank = aggregate_scores(mentions, judgments, refs)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    # Mentions with judgments (long-form JSON)
    mention_records = []
    for i, cit in enumerate(mentions):
        j = judgments.get(i)
        mention_records.append({
            "mention_index": i,
            "ref_label": cit.ref_label,
            "matched_ref_key": cit.matched_ref_key,
            "context_left": cit.context_left[-350:],   # abbreviated
            "mention": cit.mention,
            "context_right": cit.context_right[:350],
            "judgment": j.dict() if j else None
        })

    with open(f"outputs/{out_prefix}_mentions.json", "w", encoding="utf-8") as f:
        json.dump(mention_records, f, ensure_ascii=False, indent=2)

    df_rank.to_csv(f"outputs/{out_prefix}_ranked.csv", index=False)
    with open(f"outputs/{out_prefix}_ranked.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(df_rank.to_json(orient="records")), f, ensure_ascii=False, indent=2)

    print("Top inspiring references (preview):")
    print(df_rank.head(10).to_string(index=False))

    return {
        "refs": refs,
        "mentions": mentions,
        "judgments": judgments,
        "rank_df": df_rank
    }


# ============== Example usage ==============
if __name__ == "__main__":
    # ---- Replace these with your actual paper text and reference list ----
    SAMPLE_FULL_TEXT = """
    ... We build on the seminal approach by Smith et al. (2019) which introduced a
    stochastic encoder for traffic state estimation (Smith, 2019). Unlike prior work [3,4],
    our method leverages a contrastive objective to disentangle temporal factors [5–7].
    However, the assumptions in Lee (2020) do not hold in congested regimes, as noted in [8].
    This limitation motivates our hierarchical design (Kim, 2021; Park, 2022).
    """

    SAMPLE_REFERENCES = """
    [3] Doe, J. A contrastive representation for ITS. 2018. Proc. ITS.
    [4] Roe, P. Disentangling temporal factors in networks. 2019. IEEE T-ITS.
    [5] Foo, B. Temporal contrastive learning for traffic. 2020. KDD.
    [6] Bar, C. Factorization models for mobility. 2019. NeurIPS.
    [7] Baz, D. Graph disentanglement for spatiotemporal data. 2021. ICLR.
    [8] Qux, E. On failure modes in congested regimes. 2017. TR-C.
    Smith, J., Doe, P., & Lee, K. 2019. Stochastic encoders for traffic state estimation. Journal of Transport AI.
    Lee, S. 2020. Assumptions in recurrent traffic models. IEEE T-ITS.
    Kim, H. 2021. Hierarchical architectures for traffic forecasting. NeurIPS.
    Park, G. 2022. Multi-scale designs for mobility prediction. ICML.
    """

    run_pipeline(SAMPLE_FULL_TEXT, SAMPLE_REFERENCES, model_name="gpt-4o-mini")
