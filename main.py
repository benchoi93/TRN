

# 1. list all issues in Transportation Research Part C
# 2. for each issue, list all papers (doi)
# 3. for each paper, extract_inspiring_references(doi, API_KEY, INST_KEY, OPENAI_KEY)

import json
from ref_pull import extract_inspiring_references, timeit

import os
from dotenv import load_dotenv
from elsapy.elsclient import ElsClient

load_dotenv("config.env")
API_KEY = os.getenv("ELSEVIER_KEY")  # Replace with your Elsevier API key
INST_KEY = os.getenv("ELSEVIER_INSTTOKEN")  # Replace with your Elsevier Institution Token if needed
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your OpenAI API key

ISSN = "0968-090X"  # ISSN for Transportation Research Part C

client = ElsClient(api_key=API_KEY, inst_token=INST_KEY)

from elsapy.elssearch import ElsSearch
from elsapy.utils import recast_df

search = ElsSearch(f"issn({ISSN})", "scopus")
search.execute(els_client=client, get_all=True, count=5000)

import pandas as pd
df = recast_df(pd.DataFrame(search.results))
df.to_json("trc_papers.json", orient="records", lines=True)

json_path = "trc_papers.json"
import pandas as pd
df = pd.read_json(json_path, orient="records", lines=True)

# extract year (nullable)
df["year"] = (
    df["prism:coverDisplayDate"]
    .str.extract(r"(\d{4})")
    .astype(float)
    .astype("Int64")
)

# extract month name
df["month_name"] = df["prism:coverDisplayDate"].str.extract(r"([A-Za-z]+)")

# handle abbreviated ("Mar") and full ("March") month names
month_abbrev = pd.to_datetime(df["month_name"], format="%b", errors="coerce").dt.month
month_full   = pd.to_datetime(df["month_name"], format="%B", errors="coerce").dt.month

df["month"] = month_abbrev.fillna(month_full)

df = df.sort_values(by=["year", "month"], ascending=False)

df_filtered = df[df["year"].isin([2024])]

from pathlib import Path
output_dir = Path("trc_2024_inspiring_refs")
output_dir.mkdir(exist_ok=True)

from tqdm import tqdm

for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    doi = row["prism:doi"]
    inspiring_refs = extract_inspiring_references(doi, API_KEY, INST_KEY, OPENAI_KEY)
    out_filename = output_dir / (doi.replace("/", "_") + "_inspiring_refs.json")
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(inspiring_refs, f, ensure_ascii=False, indent=2)

