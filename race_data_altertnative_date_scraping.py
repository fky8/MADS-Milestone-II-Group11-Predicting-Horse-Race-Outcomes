# %% 
# Austin's initial commit
import os
import re
from datetime import date, timedelta
from io import StringIO
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
START_YEAR = 2011
END_YEAR = 2025
TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

def scrape_one_race(ds, course, race_no):
    print(f"Scraping {ds} {course} Race#{race_no} ...")
    chrome_opts = Options()
    chrome_opts.add_argument("--headless")
    chrome_opts.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_opts)
    url = (
        "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
        f"?RaceDate={ds}&Racecourse={course}&RaceNo={race_no}"
    )
    try:
        driver.get(url)
        try:
            tbl = driver.find_element(By.XPATH, TABLE_XPATH)
        except:
            return None  # Table not found
        html = tbl.get_attribute("outerHTML")
        tbls = pd.read_html(StringIO(html))
        if not tbls:
            return None
        df = tbls[0]
        df["Date"] = ds
        df["Course"] = course
        df["RaceNumber"] = race_no
        # Metadata extraction (optional, as before)
        try:
            extra_tbl = driver.find_element(By.CSS_SELECTOR, "div.race_tab table")
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
                    if "HANDICAP" in text:
                        meta_dict["Handicap"] = 1
                    if "Going :" in text:
                        meta_dict["Going"] = meta_texts[i+1].strip()
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
    finally:
        driver.quit()

script_dir = os.path.dirname(os.path.abspath(__file__))

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n===== SCRAPING YEAR {year} =====")
    all_tasks = []
    day_cursor = date(year, 1, 1)
    year_end = date(year, 12, 31)
    one_day = timedelta(days=1)
    while day_cursor <= year_end:
        ds = day_cursor.strftime("%Y/%m/%d")
        for course in ("HV", "ST"):
            # Check if Race 1 exists for this day/course
            print(f"  Checking {ds} {course} Race 1 ...")
            if scrape_one_race(ds, course, 1) is None:
                print(f"    Skipping {ds} {course} (no Race 1 found)")
                continue
            for race_no in range(1, 15):
                all_tasks.append((ds, course, race_no))
        day_cursor += one_day

    print(f"  Submitting {len(all_tasks)} races for parallel scraping...")
    dfs = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(scrape_one_race, ds, course, race_no): (ds, course, race_no) for ds, course, race_no in all_tasks}
        for i, future in enumerate(as_completed(futures), 1):
            df = future.result()
            ds, course, race_no = futures[future]
            if df is not None:
                dfs.append(df)
                print(f"    ✔ [{i}/{len(all_tasks)}] {ds} {course} Race#{race_no}: {len(df)} rows")
            else:
                print(f"    ✘ [{i}/{len(all_tasks)}] {ds} {course} Race#{race_no}: No data/table")

    if dfs:
        df_year = pd.concat(dfs, ignore_index=True)
        out_fn = f"RacePlaceData_{year}.csv"
        out_path = os.path.join(script_dir, out_fn)
        df_year.to_csv(out_path, index=False)
        print(f"\n Year {year} complete: saved {len(df_year)} rows → {out_fn}")
    else:
        print(f"\n Year {year} yielded no data, skipping file.")

print("\nAll done!")

# %%
import pandas as pd
script_dir = os.path.dirname(os.path.abspath(__file__))

combined_dfs = []
total_rows = 0
start_year = 2010
end_year = 2025

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
    
    combined_file = os.path.join(script_dir, f"RacePlaceData_modified_{start_year}_{end_year}.csv")
    df_combined.to_csv(combined_file, index=False)
    print(f"\nCombined data saved to {combined_file} with {len(df_combined)} rows")
    if len(df_combined) == total_rows:
        print("Row count matches sum of individual files.")
    else:
        print("Warning: Row count does not match sum of individual files!")
else:
    print("No data files found to combine.")
# %%
