# %% 
import os
import re
from datetime import date, timedelta
from io import StringIO

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ─── CONFIG ────────────────────────────────────────────────────────────────
START_YEAR = 2010

# Selenium:
chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=chrome_opts)

# XPath to the results table on each page
TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

# ─── DETERMINE LAST FULL YEAR ──────────────────────────────────────────────
today = date.today()
# today = date(2010, 1, 29)
LAST_FULL_YEAR = today.year - 1
END_YEAR = today.year  # includes the current year

# ─── YEAR LOOP ──────────────────────────────────────────────────────────────
for year in range(START_YEAR, END_YEAR + 1):
    print(f"\n===== SCRAPING YEAR {year} =====")
    df_year = None

    # define 1-Jan and 31-Dec for this year
    day_cursor = date(year, 1, 1)
    year_end   = date(year, 12, 31)
    one_day    = timedelta(days=1)

    # LOOP ALL DATES IN YEAR
    while day_cursor <= year_end:
        ds = day_cursor.strftime("%Y/%m/%d")
        print(f"→ Date: {ds}")

        # for each racecourse
        for course in ("HV", "ST"):
            print(f"  • Course: {course}")

            # Scraping Rule: Race 1 must exist
            url1 = (
              "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
              f"?RaceDate={ds}&Racecourse={course}&RaceNo=1"
            )
            driver.get(url1)

            try:
                tbl1 = driver.find_element(By.XPATH, TABLE_XPATH)
            except:
                print(f"    – no Race#1 → skipping {course} on {ds}")
                continue   # next course

            # start scraping
            for race_no in range(1, 15):
                url = (
                  "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
                  f"?RaceDate={ds}&Racecourse={course}&RaceNo={race_no}"
                )
                driver.get(url)
                # time.sleep(1)

                try:
                    tbl = driver.find_element(By.XPATH, TABLE_XPATH)
                except:
                    print(f"    – Race#{race_no} missing → stop this course")
                    break

                html = tbl.get_attribute("outerHTML")
                tbls = pd.read_html(StringIO(html))
                if not tbls:
                    print(f"    – Race#{race_no} empty → stop this course")
                    break

                df = tbls[0]
                df["Date"]       = ds
                df["Course"]     = course
                df["RaceNumber"] = race_no
                try:
                    extra_tbl = driver.find_element(By.CSS_SELECTOR, "div.race_tab table")
                    extra_html = extra_tbl.get_attribute("outerHTML")
                    extra_tables = pd.read_html(StringIO(extra_html))
                    if extra_tables:
                        meta_df = extra_tables[0]
                        meta_texts = meta_df.astype(str).values.flatten()
                        meta_dict = {}

                        for i, text in enumerate(meta_texts):
                            if "Class" in text and "-" in text:
                                parts = text.split(" - ")
                                meta_dict["Race type"]= parts[0]
                                meta_dict["Distance"] = parts[1].split()[0].strip()
                                meta_dict["Score range"] = parts[2].strip("()") if parts[2] else ""
                            if "Group" in text and "-" in text:
                                parts = text.split(" - ")
                                meta_dict["Race type"]= parts[0]
                                meta_dict["Distance"] = parts[1]
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
                    print("    – metadata extraction failed → continuing")
                df_year = df if df_year is None else pd.concat([df_year, df], ignore_index=True)
                print(f"    ✔ Race#{race_no}: {len(df)} rows")
        # Stop scraping if today is reached
        if day_cursor == today:
            print(f"\n Reached today's date ({today}), stopping.")
            break
            
        # next day
        day_cursor += one_day

    # after finishing the year, write out
    if df_year is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_fn = f"RacePlaceData_{year}.csv"
        out_path = os.path.join(script_dir, out_fn)
        df_year.to_csv(out_path, index=False)
        print(f"\n Year {year} complete: saved {len(df_year)} rows → {out_fn}")
    else:
        print(f"\n Year {year} yielded no data, skipping file.")

# cleanup
driver.quit()
print("\nAll done!")