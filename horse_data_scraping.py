import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

INDEX_CSV    = "hkjc_horses_index.csv"
OUTPUT_CSV   = "hkjc_horse_profiles.csv"
BASE_DOMAIN  = "https://racing.hkjc.com"


def scrape_all_horses(output_csv="hkjc_horses_index.csv"):
    """
    Scrape the HKJC 'SelectHorsebyChar' index pages for letters A–Z.
    Returns a DataFrame with: Letter, HorseName, Rating, URL.
    """
    BASE_URL = "https://racing.hkjc.com/racing/information/english/Horse/SelectHorsebyChar.aspx"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    records = []

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        resp = session.get(BASE_URL, params={"ordertype": letter})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) find both bigborder tables, and use the second one
        bigs = soup.find_all("table", class_="bigborder", width="760")
        if len(bigs) < 2:
            print(f"[{letter}] No bigborder[1], skipping")
            continue
        outer = bigs[1]

        # 2) find ALL inner tables that each hold one column of horses
        inner_tables = outer.find_all("table", attrs={"width":"100%", "border":"0"})
        if not inner_tables:
            print(f"[{letter}] No inner 100% tables, skipping")
            continue

        count = 0
        # 3) loop through every inner table + every row
        for tbl in inner_tables:
            for tr in tbl.find_all("tr"):
                # pick the <a> whose href contains Horse.aspx?HorseId=
                a = tr.find("a", href=lambda u: u and "Horse.aspx?HorseId=" in u)
                if not a:
                    continue

                name = a.get_text(strip=True)
                url  = urljoin(BASE_URL, a["href"])

                # find the rating in the same <tr>
                td_rating = tr.find("td", align="right", class_="table_eng_text")
                rating    = td_rating.get_text(strip=True).strip("()") if td_rating else ""

                records.append({
                    "Letter":    letter,
                    "HorseName": name,
                    "Rating":    rating,
                    "URL":       url
                })
                count += 1

        print(f"[{letter}] scraped {count} horses")

    # Build DataFrame
    df = pd.DataFrame(records, columns=["Letter","HorseName","Rating","URL"])
    print(f"\nDone — total {len(df)} horses scraped")
    return df


def parse_horse_profile(url: str, session: requests.Session) -> dict:
    """
    Fetch a horse page and return a dict of profile fields,
    including HorseName, HorseIndex, Country, Age, Colour, Sex, etc.
    """
    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    data = {}

    # 1) Profile title → HorseName + HorseIndex using Regex
    title_span = soup.find("span", class_="title_text")
    if title_span:
        text = title_span.get_text(strip=True)
        m = re.match(r"^(.*?)\s*\(([^)]+)\)$", text)
        if m:
            data["HorseName"]  = m.group(1).strip()
            data["HorseIndex"] = m.group(2).strip()
        else:
            data["HorseName"]  = text
            data["HorseIndex"] = ""
    else:
        data["HorseName"]  = ""
        data["HorseIndex"] = ""

    # 2) Left table (260px) → Country/Age, Colour/Sex, etc.
    left_table = soup.select_one('td[style*="width: 260px"] table.table_eng_text')
    if left_table:
        for tr in left_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            field = tds[0].get_text(" ", strip=True)
            value = tds[2].get_text(" ", strip=True)

            if field == "Country of Origin / Age" and "/" in value:
                country, age = value.split("/", 1)
                data["Country"] = country.strip()
                data["Age"]     = age.strip()
            elif field == "Colour / Sex" and "/" in value:
                colour, sex = value.split("/", 1)
                data["Colour"] = colour.strip()
                data["Sex"]    = sex.strip()
            else:
                data[field] = value

    # 3) Right table (280px) → Trainer, Owner, Rating, Sire, Dam, Dam's Sire, Same Sire
    right_table = soup.select_one('table.table_eng_text[style*="width: 280px"]')
    if right_table:
        for tr in right_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            field = tds[0].get_text(" ", strip=True)

            if field == "Same Sire":
                sel = tds[2].find("select")
                opts = [opt.get_text(strip=True) for opt in sel.find_all("option")] if sel else []
                data[field] = "|".join(opts)
            else:
                a = tds[2].find("a")
                val = a.get_text(" ", strip=True) if a else tds[2].get_text(" ", strip=True)
                data[field] = val

    # 4) Check for “PP Pre-import races footage”
    #    if any <td> exactly matches that text, flag = 1 (exist), else 0
    pp_td = soup.find("td", string=re.compile(r"^\s*PP Pre-import races footage\s*$", re.I))
    data["PP Pre-import races footage"] = 1 if pp_td else 0

    return data

def main(df_idx):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    all_records = []
    for _, row in df_idx.iterrows():
        url  = row["URL"]
        print(f"Fetching profile for {row['HorseName']} …", end=" ")
        prof = parse_horse_profile(url, session)
        print("done")

        # Merge index + profile fields
        rec = {
            "Letter":     row["Letter"],
            "HorseName":  prof.pop("HorseName", row["HorseName"]),
            "HorseIndex": prof.pop("HorseIndex", ""),
            "ProfileURL": url,
        }
        rec.update(prof)
        all_records.append(rec)

    # Build DataFrame & save CSV
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False)
    abs_path = os.path.abspath(OUTPUT_CSV)
    print(f"\nSaved {len(df)} horse profiles to {OUTPUT_CSV}")
    print(f"Full file path: {abs_path}")

if __name__ == "__main__":
    df = scrape_all_horses()
    main(df)
