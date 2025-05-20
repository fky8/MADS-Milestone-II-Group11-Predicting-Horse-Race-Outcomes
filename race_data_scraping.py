from datetime import date, timedelta
from io import StringIO

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ─── CONFIG ────────────────────────────────────────────────────────────────
START_YEAR = 2025

# Selenium: headless Chrome (change path if needed)
chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=chrome_opts)

# XPath to the results table on each page
TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

# ─── DETERMINE LAST FULL YEAR ──────────────────────────────────────────────
today = date.today()
today = date(2025, 1, 2)
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

            # scrape Race#1
            html1 = tbl1.get_attribute("outerHTML")
            tables = pd.read_html(StringIO(html1))
            if not tables:
                print(f"    – Race#1 empty → skipping {course} on {ds}")
                continue
            df = tables[0]
            df["Date"]       = ds
            df["Course"]     = course
            df["RaceNumber"] = 1
            df_year = df if df_year is None else pd.concat([df_year, df], ignore_index=True)
            print(f"    ✔ Race#1: {len(df)} rows")

            # scrape Race#2…Race#12 until one is missing
            for race_no in range(2, 13):
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
                df_year = df if df_year is None else pd.concat([df_year, df], ignore_index=True)
                print(f"    ✔ Race#{race_no}: {len(df)} rows")
        # Stop scraping if today is reached
        if day_cursor == today:
            print(f"\n Reached today's date ({today}), stopping.")
            break
            
        # next day
        day_cursor += one_day

    # after finishing the year, write it out
    if df_year is not None:
        out_fn = f"RacePlaceData_{year}.csv"
        df_year.to_csv(out_fn, index=False)
        print(f"\n Year {year} complete: saved {len(df_year)} rows → {out_fn}")
    else:
        print(f"\n Year {year} yielded no data, skipping file.")

# cleanup
driver.quit()
print("\nAll done!")