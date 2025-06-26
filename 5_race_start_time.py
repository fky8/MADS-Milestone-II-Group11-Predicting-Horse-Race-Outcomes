import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def extract_info_from_html(html: str):
    """
    Parse the HTML of the meeting reminder page to extract:
    - The meeting date
    - The total number of races
    - The meeting start time
    """
    soup = BeautifulSoup(html, "html.parser")
    for strong in soup.find_all("strong", class_="body_text_ri"):
        if "Meeting Start Time" in strong.get_text():
            text = strong.get_text(separator=" ", strip=True)
            date_part = text.split("(")[0].strip()
            races_part = text.split("races")[0].split("-")[-1].strip()
            time_part = text.split("Meeting Start Time:")[-1].strip()
            return date_part, races_part, time_part
    return None, None, None


def fetch_race_data(date_obj: datetime):
    """
    Fetch the race meeting info for a single date from the HKJC widget.
    Returns a dictionary with date, total races, and start time.
    """
    base_url = (
        "https://racing.hkjc.com/racing/english/racing-widget/meetingreminder.aspx?raceDate="
    )
    date_str = date_obj.strftime("%Y%m%d")
    try:
        response = requests.get(base_url + date_str, timeout=10)
        if response.status_code == 200:
            date_text, races, start_time = extract_info_from_html(response.text)
            if date_text and races and start_time:
                return {
                    "Date": date_text,
                    "TotalRaceNumber": races,
                    "StartTime": start_time,
                }
    except Exception as exc:
        print(f"{date_str} failed: {exc}")
    return None


def scrape_race_start_times(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Scrape start times for all race meetings between start_date and end_date (inclusive).
    Uses ThreadPoolExecutor for parallel requests.
    """
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(fetch_race_data, date_range):
            if res:
                results.append(res)
    return pd.DataFrame(results)


def process_race_place_data(
    src_csv: str = os.path.join("data", "race", "RacePlaceData_2010_2025.csv"),
    out_csv: str = os.path.join("data", "race", "race_date.csv"),
) -> pd.DataFrame:
    """
    Extracts and saves a simplified race date file with Date, Course, and RaceNumber columns.
    """
    df = pd.read_csv(src_csv)
    df = df[["Date", "Course", "RaceNumber"]].drop_duplicates()
    df = df.dropna(subset=["RaceNumber"])
    df["RaceNumber"] = df["RaceNumber"].astype(int)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce").dt.strftime("%d/%m/%Y")
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    """
    Main workflow:
    1. Scrape start times for all meetings from Jan 1, 2025 to today.
    2. Save the start times and a simplified race date file.
    """
    start_date = datetime.strptime("20250101", "%Y%m%d")
    end_date = datetime.today()
    start_times = scrape_race_start_times(start_date, end_date)
    out_start = os.path.join("data", "race", "hkjc_race_start_times.csv")
    start_times.to_csv(out_start, index=False)
    process_race_place_data()
    print("Saved hkjc_race_start_times.csv and race_date.csv")


if __name__ == "__main__":
    main()
