# %% 
# Austin's initial commit
import os
import re
from datetime import date, timedelta
from io import StringIO

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load race metadata from 2021 to 2025 files
metadata_dfs = []
for year in range(2024, 2026):
    file_path = os.path.join(script_dir, f"RacePlaceData_{year}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        metadata_dfs.append(df[["Date", "Course", "RaceNumber"]])
    else:
        print(f"File {file_path} does not exist, skipping.")

if not metadata_dfs:
    print("No metadata files found for years 2021–2025. Exiting.")
    exit()

race_meta_df = (
    pd.concat(metadata_dfs)
    .drop_duplicates()
    .dropna(subset=["Date", "Course", "RaceNumber"])
)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from threading import local
from concurrent.futures import ThreadPoolExecutor, as_completed

driver_holder = local()

def get_driver():
    if not hasattr(driver_holder, "driver"):
        driver_holder.driver = webdriver.Chrome(options=chrome_opts)
    return driver_holder.driver

# ─── CONFIG ────────────────────────────────────────────────────────────────
START_YEAR = 2017

chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")

print("\n===== SCRAPING FROM METADATA FILE =====")

def scrape_race(row):
    ds = row.Date
    course = row.Course
    race_no = int(row.RaceNumber)
    url = (
        "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
        f"?RaceDate={ds}&Racecourse={course}&RaceNo={race_no}"
    )
    try:
        local_driver = get_driver()
        local_driver.get(url)
        tbl = local_driver.find_element(By.XPATH, '//*[@id="innerContent"]/div[2]/div[5]/table')
        html = tbl.get_attribute("outerHTML")
        tbls = pd.read_html(StringIO(html))
        if not tbls:
            return None

        df = tbls[0]
        df["Date"] = ds
        df["Course"] = course
        df["RaceNumber"] = race_no

        try:
            extra_tbl = local_driver.find_element(By.CSS_SELECTOR, "div.race_tab table")
            extra_html = extra_tbl.get_attribute("outerHTML")
            extra_tables = pd.read_html(StringIO(extra_html))
            if extra_tables:
                meta_df = extra_tables[0]
                meta_texts = meta_df.astype(str).values.flatten()
                meta_dict = {}
                for i, text in enumerate(meta_texts):
                    if re.match(r'.*- \d{4,}M.*', text):
                        parts = text.split(" - ")
                        meta_dict["Race type"]= parts[0]
                        meta_dict["Distance"] = parts[1]
                        if len(parts)>2:
                            meta_dict["Score range"] = parts[2].strip("()")
                    if "Going :" in text:
                        meta_dict["Going"] = meta_texts[i+1].strip()
                    if "HANDICAP" in text:
                        meta_dict["Handicap"] = 1
                    if "Course :" in text:
                        meta_dict["Course Detail"] = meta_texts[i+1].strip()
                    if text == "Time :":
                        times = [t.strip("()") for t in meta_texts[i+1:i+7] if t.startswith("(")]
                        for idx, val in enumerate(times, start=1):
                            meta_dict[f"Time {idx}"] = val
                        for idx in range(len(times)+1, 7):
                            meta_dict[f"Time {idx}"] = float("nan")
                    if "Sectional Time" in text:
                        sects = [meta_texts[i+j].strip() for j in range(1, 7) if i+j < len(meta_texts) and meta_texts[i+j].strip()]
                        for idx, val in enumerate(sects, start=1):
                            meta_dict[f"Sectional Time {idx}"] = val.split()[0]
                        for idx in range(len(sects)+1, 7):
                            meta_dict[f"Sectional Time {idx}"] = float("nan")
                for key, val in meta_dict.items():
                    df[key] = val
        except:
            pass
        return df
    except Exception as e:
        print(f"    – Race#{race_no} ({ds} {course}) failed: {e}")
        return None

from collections import defaultdict
from datetime import datetime

df_by_year = defaultdict(list)
years_completed = set()
last_printed_date = ""

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(scrape_race, row): row for _, row in race_meta_df.iterrows()}
    for future in as_completed(futures):
        row = futures[future]
        ds = row["Date"]
        course = row["Course"]
        race_no = int(row["RaceNumber"])
        year = ds.split("/")[0]
        if ds != last_printed_date:
            print(f"Date: {ds}")
            print(f"  • Course: {course}")
            last_printed_date = ds
        try:
            result = future.result()
            if result is not None:
                df_by_year[year].append(result)
                print(f"    ✔ Race#{race_no}: {len(result)} rows")
                # Check if all races for this year are collected
                collected_dates = set(
                    d["Date"].iloc[0]
                    for d in df_by_year[year]
                    if "Date" in d.columns and not pd.isnull(d["Date"].iloc[0])
                )
                expected_dates = set(
                    race_meta_df[
                        race_meta_df["Date"].notna() & race_meta_df["Date"].str.startswith(year)
                    ]["Date"].unique()
                )
                if year not in years_completed and collected_dates >= expected_dates:
                    output_path = os.path.join(script_dir, f"RacePlaceData_scraped_{year}.csv")
                    pd.concat(df_by_year[year], ignore_index=True).to_csv(output_path, index=False)
                    print(f"\nYear {year} complete: saved {sum(len(df) for df in df_by_year[year])} rows → {output_path}")
                    years_completed.add(year)
            else:
                print(f"    – Race#{race_no} missing or failed → skipping")
        except Exception as e:
            print(f"    – Race#{race_no} ({ds} {course}) exception: {e}")

# Close thread-local driver if exists
if hasattr(driver_holder, "driver"):
    driver_holder.driver.quit()

# %%
import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))

combined_dfs = []
total_rows = 0
start_year = 2010
end_year = 2016

for year in range(start_year, end_year + 1):
    file_path = os.path.join(script_dir, f"RacePlaceData_{year}.csv")
    if os.path.exists(file_path):
        df_year = pd.read_csv(file_path)
        row_count = len(df_year)
        print(f"Loaded {file_path} with {row_count} rows")
        combined_dfs.append(df_year)
        total_rows += row_count
    else:
        print(f"File {file_path} does not exist, skipping.")

if combined_dfs:
    # Concatenate dataframes
    df_combined = pd.concat(combined_dfs, ignore_index=True)
    
    # Clean up columns
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('Unnamed')]
    df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('RACE \d+')]
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    
    combined_file = os.path.join(script_dir, f"RacePlaceData_{start_year}_{end_year}.csv")
    df_combined.to_csv(combined_file, index=False)
    print(f"\nCombined data saved to {combined_file} with {len(df_combined)} rows")
    if len(df_combined) == total_rows:
        print("Row count matches sum of individual files.")
    else:
        print("Warning: Row count does not match sum of individual files!")
else:
    print("No data files found to combine.")
# %%
