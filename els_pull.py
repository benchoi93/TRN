import json
import httpx
from dotenv import load_dotenv
import os

load_dotenv("config.env")
API_KEY = os.getenv("ELSEVIER_KEY")  # Replace with your Elsevier API key

def scopus_fulltext_via_ip(doi: str, apikey: str):
    """
    Try to retrieve full-text JSON from Elsevier Article API using institutional IP entitlement.
    Works only if your public IP is recognized as subscribed.
    """
    headers = {
        "X-ELS-APIKey": apikey,
        "Accept": "application/json"
    }

    params = {"view": "FULL"}  # ensure FULL view is requested
    timeout = httpx.Timeout(10.0, connect=60.0)

    with httpx.Client(timeout=timeout, headers=headers) as client:
        url = f"https://api.elsevier.com/content/article/doi/{doi}"
        r = client.get(url, params=params)

        # --- Debug information ---
        print(f"HTTP status: {r.status_code}")
        print(f"Entitlement header: {r.headers.get('X-ELS-Entitlement-Status')}")
        if r.status_code != 200:
            print("Response text (first 500 chars):")
            print(r.text[:500])
            r.raise_for_status()

        return r.json()
        


def scopus_ref_via_ip(doi: str, apikey: str):
    """
    Try to retrieve full-text JSON from Elsevier Article API using institutional IP entitlement.
    Works only if your public IP is recognized as subscribed.
    """
    headers = {
        "X-ELS-APIKey": apikey,
        "Accept": "application/json"
    }

    params = {"view": "REF"}  # ensure REF view is requested
    # params = { }  # ensure REF view is requested
    timeout = httpx.Timeout(10.0, connect=60.0)

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
    
def extract_references(data):
    """
    Extracts structured reference list (authors, year, title, full text) from Elsevier Article API JSON.
    Returns list of dicts.
    """
    try:
        refs = data["full-text-retrieval-response"]["references"]["reference"]
    except KeyError:
        print("⚠️ No references found in response.")
        return []

    parsed_refs = []
    for r in refs:
        refinfo = r.get("ref-info", {})
        title = refinfo.get("ref-title", {}).get("ref-titletext", "")
        year = refinfo.get("ref-publicationyear", {}).get("@first", "")
        authors = []
        for a in refinfo.get("ref-authors", {}).get("author", []):
            name = a.get("ce:indexed-name") or a.get("ce:surname")
            if name:
                authors.append(name)
        fulltext = refinfo.get("ref-fulltext", "")
        ref_id = r.get("ref-id", {}).get("@id", "")

        parsed_refs.append({
            "ref_id": ref_id,
            "title": title,
            "authors": ", ".join(authors),
            "year": year,
            "fulltext": fulltext
        })

    return parsed_refs


# === Example usage ===
if __name__ == "__main__":
    # doi = "10.1016/j.trc.2021.103091"   # Non-OA example
    doi = "10.1016/j.tre.2023.103213" # OA example
    data = scopus_fulltext_via_ip(doi, API_KEY)

    # --- Access fields ---
    core = data["full-text-retrieval-response"]["coredata"]
    print("\nTitle:", core.get("dc:title"))
    print("Abstract:", core.get("dc:description"))

    fulltext = data["full-text-retrieval-response"].get("originalText")
    if fulltext:
        print("\nFull text snippet:\n", fulltext[:500])
    else:
        print("\n⚠️ No 'originalText' returned — likely not entitled via current IP.")

    references = scopus_ref_via_ip(doi, API_KEY)

    # === Reference list ===
    refs = extract_references(data)
    print(f"\nExtracted {len(refs)} references:")
    for r in refs[:5]:  # show first 5
        print(f"- {r['authors']} ({r['year']}): {r['title']}")
        print(f"  {r['fulltext']}\n")