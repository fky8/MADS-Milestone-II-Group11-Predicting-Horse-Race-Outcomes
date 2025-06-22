import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from collections import defaultdict

# ── Directory setup ───────────────────────────────────────────────
DATA_DIR = "data"
RACE_DIR = os.path.join(DATA_DIR, "race")
COMMENTS_DIR = os.path.join(DATA_DIR, "comments_by_year")

# ── Gear codes and modifiers ──────────────────────────────────────
gear_codes = [
    "B","BO","CC","CP","CO","E","H","P","PC","PS",
    "SB","SR","TT","V","VO","XB"
]

modifier_map = {
    "1": "first_time",
    "2": "replaced",
    "-": "removed",
}

# Pre-compile regex for gear parsing
pattern = re.compile(r"^([A-Z]+?)([12\-]?)$")

# ── Gear expansion function ───────────────────────────────────────
def expand_gear(s: str) -> dict:
    """
    Given a Gear string like "H1/SR-", returns a dict with keys:
      - For each code X in gear_codes: X (0/1), X_first_time (0/1), X_replaced, X_removed
    This allows for easy one-hot encoding of gear usage and changes.
    """
    parts = s.split("/") if isinstance(s, str) else []
    out = {}

    # Initialize all gear flags to 0
    for code in gear_codes:
        out[code]              = 0
        out[f"{code}_first_time"] = 0
        out[f"{code}_replaced"]   = 0
        out[f"{code}_removed"]    = 0

    # Parse each slash‐delimited chunk
    for part in parts:
        m = pattern.match(part.strip())
        if not m:
            continue
        code, mod = m.group(1), m.group(2)
        if code not in gear_codes:
            continue

        if mod == "-":
            # Gear was removed this run
            out[f"{code}_removed"] = 1
            out[code] = 0
        else:
            # Gear is present
            out[code] = 1
            # Mark if first_time or replaced
            if mod in modifier_map:
                out[f"{code}_{modifier_map[mod]}"] = 1

    return out

def prepare_race_date_csv(
    source_csv=os.path.join(RACE_DIR, "RacePlaceData_2010_2025.csv"),
    output_csv=os.path.join(RACE_DIR, "race_date.csv"),
):
    """
    Create a CSV containing only Date, Course and RaceNumber.
    If output_csv already exists it will be left untouched.
    Returns the DataFrame that is written to disk.
    """
    if os.path.exists(output_csv):
        return pd.read_csv(output_csv)

    cols = ["Date", "Course", "RaceNumber"]
    df = pd.read_csv(source_csv, usecols=cols, low_memory=False)
    df = df.dropna(subset=cols)
    df["RaceNumber"] = pd.to_numeric(df["RaceNumber"], errors="coerce").astype("Int64")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "RaceNumber", "Course"])
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")
    df = df.drop_duplicates()
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} rows to {output_csv}")
    return df

def extract_unique_dates(master_csv=os.path.join(RACE_DIR, "race_date.csv")):
    """
    Reads master CSV, parses Date column as day-first, and returns sorted unique dates as list of Timestamps.
    """
    df = pd.read_csv(
        master_csv,
        parse_dates=["Date"],
        dayfirst=True,
        low_memory=False
    )
    unique_dates = (
        pd.to_datetime(df["Date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return unique_dates

def scrape_comments_for_dates(dates, max_rows=None):
    """
    Scrape CORunning.aspx for RaceNo=1..13 on each date in `dates`.
    If max_rows is set, stop once we've collected >= max_rows total rows.
    Returns a single concatenated DataFrame.
    """
    base_url   = "https://racing.hkjc.com/racing/information/English/Reports/CORunning.aspx"
    all_frames = []
    total_rows = 0

    for dt in dates:
        date_str = dt.strftime("%Y%m%d")
        for race_no in range(1, 14):
            # Request the comments page for this date and race number
            resp = requests.get(base_url, params={"Date": date_str, "RaceNo": race_no})
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            tbl  = soup.find("table", class_="table_bd")
            if tbl is None:
                continue

            # Parse the HTML table into a DataFrame
            df = pd.read_html(StringIO(str(tbl)))[0]
            if df.empty:
                continue

            df["Date"]       = dt.strftime("%Y-%m-%d")
            df["RaceNumber"] = race_no
            all_frames.append(df)

            total_rows += len(df)
            print(f"[{dt} Race {race_no}] +{len(df)} rows → total {total_rows}")

            if max_rows is not None and total_rows >= max_rows:
                print(f"Reached {total_rows} rows (>= {max_rows}), stopping early.")
                return pd.concat(all_frames, ignore_index=True)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

def scrape_comments_by_years(unique_dates, years=None):
    """
    Given a list of unique dates, scrape comments grouped by year for specified years.
    Returns a dict of year -> DataFrame.
    """
    if years is None:
        years = list(set(d.year for d in unique_dates))
    dates_by_year = defaultdict(list)
    for d in unique_dates:
        if d.year in years:
            dates_by_year[d.year].append(d)

    year_dfs = {}
    for year in sorted(dates_by_year.keys()):
        print(f"\n📅 Scraping for year {year} with {len(dates_by_year[year])} dates...")
        df_year_comments = scrape_comments_for_dates(dates_by_year[year], max_rows=None)
        year_dfs[year] = df_year_comments
    return year_dfs

def combine_comments_csv(start_year=None, end_year=None, output_file=os.path.join(DATA_DIR, "comments_combined.csv")):
    """
    Combine all yearly comment CSV files into a single DataFrame and save as a new CSV.
    """
    dfs = []
    comment_dir = COMMENTS_DIR
    files = [f for f in os.listdir(comment_dir) if f.startswith("comments_") and f.endswith(".csv")]
    years = sorted(int(re.search(r"\d{4}", f).group()) for f in files if re.search(r"\d{4}", f))

    if not years:
        print("No valid comment files found.")
        return pd.DataFrame()

    if start_year is None:
        start_year = years[0]
    if end_year is None:
        end_year = years[-1]

    for year in range(start_year, end_year + 1):
        fn = os.path.join(comment_dir, f"comments_{year}.csv")
        if os.path.exists(fn):
            print(f"Reading {fn}...")
            df = pd.read_csv(fn)
            dfs.append(df)
        else:
            print(f"File {fn} not found, skipping.")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
        combined.to_csv(output_file, index=False)
        print(f"\n✅ Combined {len(dfs)} files into {len(combined)} rows → {output_file}")
        return combined
    else:
        print("No files found to combine.")
        return pd.DataFrame()

def clean_comments_file(file):
    """
    Remove rows with unavailable comments from the comments CSV file.
    """
    df = pd.read_csv(file)
    df = df[~df['Comment'].str.contains("Comments on Running is not available for this race.", na=False)]
    df.to_csv(file, index=False)
    print(f"✅ Cleaned and saved in-place to {file}. Remaining rows: {len(df)}.")

def expand_gear_columns(input_file=os.path.join(DATA_DIR, "comments_2010_to_2025_combined.csv")):
    """
    Expand the Gear column into multiple binary columns for each gear code and modifier.
    """
    df = pd.read_csv(input_file)
    expanded = df["Gear"].apply(expand_gear).apply(pd.Series)
    df_expanded = pd.concat([df, expanded], axis=1)
    df_expanded.to_csv(input_file, index=False)
    print(f"✅ Expanded gear columns and saved in-place to {input_file}")

def main():
    """
    Main workflow:
    1. Prepare race_date.csv if needed.
    2. Extract unique race dates.
    3. Scrape comments for each year and save.
    4. Combine and clean all comments.
    5. Expand gear columns for modeling.
    """
    # Build race_date.csv from the master race results if needed
    prepare_race_date_csv()

    # Extract unique dates from master CSV
    unique_dates = extract_unique_dates()

    # Dynamically determine years from unique_dates
    years = sorted(set(d.year for d in unique_dates))
    start_year, end_year = years[0], years[-1]

    # Scrape comments by year
    year_dfs = scrape_comments_by_years(unique_dates, years=years)
    for year, df_year in year_dfs.items():
        out_fn = os.path.join(COMMENTS_DIR, f"comments_{year}.csv")
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        df_year.to_csv(out_fn, index=False)
        print(f"✅ Saved {len(df_year)} rows to {out_fn}")

    # Combine and clean comments
    combined_file = os.path.join(DATA_DIR, f"comments_{start_year}_to_{end_year}_combined.csv")
    combine_comments_csv(start_year=start_year, end_year=end_year, output_file=combined_file)
    clean_comments_file(combined_file)

    # Expand gear columns
    expand_gear_columns(input_file=combined_file)

if __name__ == "__main__":
    main()
