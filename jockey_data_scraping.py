import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from io import StringIO

OUTPUT_CSV   = "hkjc_horse_profiles.csv"
BASE_DOMAIN  = "https://racing.hkjc.com"

TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[3]'

TABLE_XPATH_ERROR = '//*[@id="innerContent"]/div[2]/div[1]'

# TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

def get_jockey_name_list():
    """
    Gets a list of jockey names to scrape from the HKJC website.
    Returns a list of jockey names.
    """
    df_record = pd.read_excel('./data/jockey_names.xlsx')
    return df_record['jockey_name'].tolist()

def get_jockey_id_list():
    """
    Gets a list of jockey names to scrape from the HKJC website.
    Returns a list of jockey names.
    """
    df_record = pd.read_excel('./data/jockey_names.xlsx')
    return df_record['jockey_id'].tolist()

SEASONS = [
    "Current", 
    "Previous"
]

BASE_URL = "https://racing.hkjc.com/racing/information/English/Jockey/JockeyPastRec.aspx"


def scrape_all_jockeys():
    """
    Scrape the HKJC 'SelectJockeybyId' index pages for letters A–Z.
    Returns a DataFrame with: Letter, Jockey Name, Rating, URL.
    """

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    
    df_list = []
    jockey_list = get_jockey_id_list()

    for jockey in jockey_list:
        for season in SEASONS:
            is_last_page = False
            min_race_index = -1
            page_index = 1
            while is_last_page is False:

                # print(f"JockeyId{jockey} season {season} page {page_index}")

                url = f"{BASE_URL}?JockeyId={jockey}&Season={season}&PageNum={page_index}"
                try:
                    driver.get(url)
                except:
                    print(f"    JockeyId{jockey} missing → next jockey")
                    break

                time.sleep(1)
                
                try:
                    tbl = driver.find_element(By.CLASS_NAME, 'ridingRec')
                except:
                    print(f"    JockeyId{jockey} missing → next jockey")
                    break


                try:
                    tbl = driver.find_element(By.CLASS_NAME, 'ridingRec')
                    html = tbl.get_attribute("outerHTML")
                    if 'errorour' in html:
                        print(f"    Jockey#{jockey} No Error")
                        break
                except:
                    continue
                    

                html = tbl.get_attribute("outerHTML")
                tbls = pd.read_html(StringIO(html))
                
                # print(url)
                df = tbls[0]
                df['jockey_id'] = jockey
                df['season'] = season

                min_race_index_next = int(df['Race Index'].iloc[-1]) 


                if min_race_index == min_race_index_next:
                    is_last_page = True
                    # print(f"    JockeyId{jockey} last page {page_index}") 
                else:
                    df_list.append(df)
                    min_race_index = min_race_index_next 
                    # print(f"    JockeyId{jockey} page {page_index} min_race_index {min_race_index}")     


                page_index += 1

    if len(df_list) > 0:
        df_concat = pd.concat(df_list, ignore_index=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"jockey_race_data.csv")
        df_concat.to_csv(file_path, index=False)
        print(len(df_concat))
    else:
        print("No data to save.")



    # Build DataFrame
    driver.quit()
    # df = pd.DataFrame(records, columns=["Letter","HorseName","Rating","URL"])
    # print(f"\nDone — total {len(df)} horses scraped")
    # return df


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
# %%
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
            "HorseName":  prof.pop("HorseName", row["HorseName"]),
            "HorseIndex": prof.pop("HorseIndex", ""),
            "ProfileURL": url,
        }
        rec.update(prof)
        all_records.append(rec)

    # Build DataFrame & drop unneeded columns as early as possible
    df = pd.DataFrame(all_records)
    df.columns = df.columns.str.replace('*', '', regex=False)
    print (df.columns)
    df = df.replace(r'\$', '', regex=True)
    # Drop unneeded columns as early as possible
    df.drop(columns=["ProfileURL"], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    abs_path = os.path.abspath(OUTPUT_CSV)
    print(f"\nSaved {len(df)} horse profiles to {OUTPUT_CSV}")
    print(f"Full file path: {abs_path}")

if __name__ == "__main__":
    # df = scrape_all_horses()
    # main(df)
    scrape_all_jockeys()


# %%
