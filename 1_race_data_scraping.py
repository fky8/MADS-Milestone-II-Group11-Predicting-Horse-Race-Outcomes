# %%
import os
import re
from datetime import date, timedelta
from io import StringIO
from multiprocessing import Pool

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ─── CONFIG ────────────────────────────────────────────────────────────────
START_YEAR = 2010
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "race")
YEAR_DIR = os.path.join(DATA_DIR, "race_result_by_year")

# Selenium setup function
def create_driver():
    """
    Create and return a headless Chrome WebDriver instance.
    """
    chrome_opts = Options()
    chrome_opts.add_argument("--headless")
    chrome_opts.add_argument("--disable-gpu")
    return webdriver.Chrome(options=chrome_opts)

# XPath to the results table on each page
TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

# ─── SCRAPING FUNCTIONS ────────────────────────────────────────────────────

def scrape_year(year, driver, today):
    """
    Scrape all race data for a given year using the provided Selenium driver.
    Iterates over each day and both courses (HV, ST), and for each race on that day,
    extracts the results table and relevant metadata.
    """
    print(f"\n===== SCRAPING YEAR {year} =====")
    df_year = None

    day_cursor = date(year, 1, 1)
    year_end = date(year, 12, 31)
    one_day = timedelta(days=1)

    while day_cursor <= year_end:
        ds = day_cursor.strftime("%Y/%m/%d")
        print(f"→ Date: {ds}")

        for course in ("HV", "ST"):
            print(f"  • Course: {course}")

            # Check if Race 1 exists for this course and date
            url1 = (
                "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
                f"?RaceDate={ds}&Racecourse={course}&RaceNo=1"
            )
            driver.get(url1)

            try:
                driver.find_element(By.XPATH, TABLE_XPATH)
            except Exception:
                print(f"    – no Race#1 → skipping {course} on {ds}")
                continue

            # Scrape all races for this course and date
            for race_no in range(1, 15):
                url = (
                    "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
                    f"?RaceDate={ds}&Racecourse={course}&RaceNo={race_no}"
                )
                driver.get(url)

                try:
                    tbl = driver.find_element(By.XPATH, TABLE_XPATH)
                except Exception:
                    print(f"    – Race#{race_no} missing → stop this course")
                    break

                html = tbl.get_attribute("outerHTML")
                tbls = pd.read_html(StringIO(html))
                if not tbls:
                    print(f"    – Race#{race_no} empty → stop this course")
                    break

                df = tbls[0]
                df["Date"] = ds
                df["Course"] = course
                df["RaceNumber"] = race_no

                # Try to extract additional metadata from the page
                try:
                    extra_tbl = driver.find_element(By.CSS_SELECTOR, "div.race_tab table")
                    extra_html = extra_tbl.get_attribute("outerHTML")
                    extra_tables = pd.read_html(StringIO(extra_html))
                    if extra_tables:
                        meta_df = extra_tables[0]
                        meta_texts = meta_df.astype(str).values.flatten()
                        meta_dict = {}

                        for i, text in enumerate(meta_texts):
                            if re.match(r".*- \d{4,}M.*", text):
                                parts = text.split(" - ")
                                meta_dict["Race type"] = parts[0]
                                meta_dict["Distance"] = parts[1]
                                if len(parts) > 2:
                                    meta_dict["Score range"] = parts[2].strip("()")
                            if "HANDICAP" in text:
                                meta_dict["Handicap"] = 1
                            if "Going :" in text:
                                meta_dict["Going"] = meta_texts[i + 1].strip()
                            if "Course :" in text:
                                meta_dict["Course Detail"] = meta_texts[i + 1].strip()
                            if text == "Time :":
                                times = [t.strip("()") for t in meta_texts[i + 1 : i + 7] if t.startswith("(")]
                                for idx, val in enumerate(times, start=1):
                                    meta_dict[f"Time {idx}"] = val
                                for idx in range(len(times) + 1, 7):
                                    meta_dict[f"Time {idx}"] = float("nan")
                            if "Sectional Time" in text:
                                sects = [meta_texts[i + j].strip() for j in range(1, 7) if i + j < len(meta_texts) and meta_texts[i + j].strip()]
                                for idx, val in enumerate(sects, start=1):
                                    meta_dict[f"Sectional Time {idx}"] = val.split()[0]
                                for idx in range(len(sects) + 1, 7):
                                    meta_dict[f"Sectional Time {idx}"] = float("nan")
                        for key, val in meta_dict.items():
                            df[key] = val
                except Exception:
                    print("    – metadata extraction failed → continuing")

                # Concatenate this race's data to the year's DataFrame
                df_year = df if df_year is None else pd.concat([df_year, df], ignore_index=True)
                print(f"    ✔ Race#{race_no}: {len(df)} rows")

        if day_cursor == today:
            print(f"\n Reached today's date ({today}), stopping.")
            break

        day_cursor += one_day

    return df_year

def combine_yearly_files(start_year=START_YEAR, end_year=None):
    """
    Combine all yearly CSV files into a single DataFrame, clean up columns,
    and save the combined result as a new CSV.
    """
    if end_year is None:
        end_year = date.today().year
    os.makedirs(YEAR_DIR, exist_ok=True)
    combined_dfs = []
    total_rows = 0
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(YEAR_DIR, f"RacePlaceData_{year}.csv")
        if os.path.exists(file_path):
            df_year = pd.read_csv(file_path)
            row_count = len(df_year)
            print(f"Loaded {file_path} with {row_count} rows")
            combined_dfs.append(df_year)
            total_rows += row_count
        else:
            print(f"File {file_path} does not exist, skipping.")

    if combined_dfs:
        df_combined = pd.concat(combined_dfs, ignore_index=True)
        # Remove unwanted columns and duplicates
        df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('Unnamed')]
        df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('RACE \d+')]
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        phantom_cols = [
            "('Dividend', 'Pool')", "('Dividend', 'Winning Combination')",
            "('Dividend', 'Dividend (HK$)')", "('Date', '')",
            "('Course', '')", "('RaceNumber', '')",
        ]
        df_combined = df_combined.drop(columns=[col for col in df_combined.columns if str(col) in phantom_cols], errors='ignore')
        # Remove rows with missing or blank horse names
        df_combined = df_combined[df_combined['Horse'].notna() & (df_combined['Horse'].astype(str).str.strip() != '')]
        combined_file = os.path.join(DATA_DIR, f"RacePlaceData_{start_year}_{end_year}.csv")
        df_combined.to_csv(combined_file, index=False)
        print(f"\nCombined data saved to {combined_file} with {len(df_combined)} rows")
        if len(df_combined) == total_rows:
            print("Row count matches sum of individual files.")
        else:
            print("Warning: Row count does not match sum of individual files!")
    else:
        print("No data files found to combine.")

def scrape_year_wrapper(year):
    """
    Wrapper function for multiprocessing: creates a driver, scrapes a year,
    saves the result, and closes the driver.
    """
    today = date.today()
    driver = create_driver()
    df_year = scrape_year(year, driver, today)
    driver.quit()
    if df_year is not None:
        os.makedirs(YEAR_DIR, exist_ok=True)
        out_fn = os.path.join(YEAR_DIR, f"RacePlaceData_{year}.csv")
        df_year.to_csv(out_fn, index=False)
        print(f"\n Year {year} complete: saved {len(df_year)} rows → {out_fn}")
    else:
        print(f"\n Year {year} yielded no data, skipping file.")

def main():
    """
    Main function: runs the scraping in parallel for each year,
    then combines all yearly files into a single CSV.
    """
    today = date.today()
    years = list(range(START_YEAR, today.year + 1))
    with Pool(processes=4) as pool:
        pool.map(scrape_year_wrapper, years)
    print("\nAll done!")
    combine_yearly_files(start_year=START_YEAR, end_year=today.year)

if __name__ == "__main__":
    main()
