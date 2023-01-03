import json
import shutil
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests
from rich import print
from sqlite_utils import Database

from .catalog import is_environmental
from .chi import ChiAnalysis


def fetch_and_cache_url(url: str, params: dict = {}) -> dict[Any, Any]:
    """
    Fetch a URL and cache the response in data/private
    """

    url_with_params = url + "?" + urlencode(params, doseq=True)
    parsed = urlparse(url_with_params)
    filename = Path(
        "data",
        "interim",
        "cache",
        parsed.netloc,
        parsed.path.lstrip("/") + urlencode(params, doseq=True) + ".json",
    )
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists() is False:
        r = requests.get(url, params=params)
        assert r.status_code == 200, f"{r.url} returned status code {r.status_code}"
        print(f"Downloading {r.url}")
        data = r.json()
        with open(filename, "w") as f:
            json.dump(data, f)
        sleep(1)
    else:
        with open(filename) as f:
            data = json.load(f)
    return data


def fetch_petition_names_based_on_keywords():
    """
    Given a set of keywords, query the API for petitions matching those keywords
    """
    keywords = []

    keywords = pd.read_csv(Path("data", "raw", "keywords.csv"), header=None)
    keywords = keywords[0].str.strip().tolist()

    print(
        f"Fetching petitions based on [blue]{len(keywords)}[/blue] keywords from keywords.csv: {', '.join(keywords)}"
    )

    petitions = []

    for keyword in keywords:
        count_for_keyword = 0
        next_page = 1
        while next_page > 0:

            j = fetch_and_cache_url(
                "https://petition.parliament.uk/petitions.json",
                params={"q": keyword, "page": next_page},
            )

            for p in j["data"]:
                count_for_keyword += 1
                petition = extract_petition_data(p)
                petitions.append(petition)

            if j["links"]["next"]:
                next_page = get_page_number_from_url(j["links"]["next"])
            else:
                next_page = 0

        print(
            f"Found [blue]{count_for_keyword}[/blue] petitions for keyword [blue]{keyword}[/blue]"
        )

    df = pd.DataFrame(petitions)
    # remove duplicates of id
    df = df.drop_duplicates(subset="id")
    # print how many petitions were found
    print(f"Found [blue]{len(df)}[/blue] petitions")
    df.to_csv(Path("data", "interim", "petitions.csv"), index=False)


def ml_classifer():
    """
    Run the ML classifier on the petitions
    """

    print("[green]Running ML classifier[/green]")

    df = pd.read_csv(Path("data", "interim", "petitions.csv"))

    # highest signature_count to top
    df = df.sort_values(by="signature_count", ascending=False)

    # limit to petitions with more than 1000 signatures
    df = df[df["signature_count"] > 1000]

    # constructed joined action and background columns
    full_text = df["action"] + " " + df["background"]

    # run environment check on full_text
    results = is_environmental(full_text.tolist())

    df["is_environmental"] = [result["result"] for result in results]
    df["openai_explanation"] = [result["explanation"] for result in results]
    df.to_csv(Path("data", "interim", "petitions.csv"), index=False)


def fetch_constituency_values():
    """
    For each petition with more than 10,000 signatures, fetch the constituency level data
    """

    print("[green]Fetching constituency level data[/green]")

    petitions = pd.read_csv(Path("data", "interim", "petitions.csv"))
    large_petitions = petitions[petitions["signature_count"] > 1000].to_dict("records")

    all_sigs = []

    for petition in large_petitions:
        j = fetch_and_cache_url(petition["url"])
        for cons in j["data"]["attributes"]["signatures_by_constituency"]:
            sigs = {
                "petition_id": petition["id"],
                "constituency": cons["name"],
                "gss": cons["ons_code"],
                "signatures": cons["signature_count"],
            }
            all_sigs.append(sigs)

    df = pd.DataFrame(all_sigs)
    df.to_csv(Path("data", "interim", "constituency_signatures.csv"), index=False)


def extract_petition_data(p: dict) -> dict[str, str]:
    """
    Extract a flat dictionary of petition data from the JSON returned by the API
    """
    petition = {
        "id": p["id"],
        "url": p["links"]["self"],
        "state": p["attributes"]["state"],
        "action": p["attributes"]["action"],
        "background": p["attributes"]["background"],
        "additional_details": p["attributes"]["additional_details"],
        "signature_count": p["attributes"]["signature_count"],
        "date_created": p["attributes"]["created_at"],
    }

    try:
        petition["date_responded"] = p["attributes"]["government_response"][
            "responded_on"
        ]
    except TypeError:
        petition["date_responded"] = None

    try:
        petition["date_debated"] = p["attributes"]["debate"]["debated_on"]
    except TypeError:
        petition["date_debated"] = None

    return petition


def process_significance():
    """
    Looking at the counts for each constituency, add if it is significant or not
    based on a chi square analysis
    """
    print("[green]Calculating significance[/green]")
    con_df = pd.read_csv(Path("data", "interim", "constituency_signatures.csv"))
    grid = con_df.pivot_table(
        index="petition_id", columns="gss", values="signatures", aggfunc="sum"
    ).fillna(0)

    chi = ChiAnalysis(grid)
    g = grid.reset_index().melt(id_vars="petition_id", value_name="signatures")
    r = chi.resid.reset_index().melt(id_vars="petition_id", value_name="std.res")
    e = chi.expected.reset_index().melt(id_vars="petition_id", value_name="expected")
    chi_df = g.merge(r).merge(e)
    # add a columm saying if the resid is significant
    chi_df["significant"] = chi_df["std.res"].abs() > chi.significance

    # sort by highest number of signatures, and constituency
    chi_df = chi_df.sort_values(by=["petition_id", "signatures"], ascending=False)

    chi_df.to_csv(
        Path("data", "interim", "constituency_signatures_with_significance.csv"),
        index=False,
    )


def get_page_number_from_url(url: str):
    """
    Given the url, extract the page parameter
    """
    u = urlparse(url)
    q = parse_qs(u.query)
    return int(q["page"][0])


def move_to_package():
    """
    Move the interim files to the package
    """
    print("[green]Moving files to package[/green]")
    shutil.copy(
        Path("data", "interim", "petitions.csv"),
        Path("data", "packages", "environmental_petitions", "petitions.csv"),
    )
    shutil.copy(
        Path("data", "interim", "constituency_signatures_with_significance.csv"),
        Path("data", "packages", "environmental_petitions", "signatures.csv"),
    )


def build_database():
    fetch_petition_names_based_on_keywords()
    ml_classifer()
    fetch_constituency_values()
    process_significance()
    move_to_package()


if __name__ == "__main__":
    build_database()
