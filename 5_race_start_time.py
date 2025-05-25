import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import time
import concurrent.futures
# %%
def extract_info_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    strongs = soup.find_all("strong", class_="body_text_ri")
    for s in strongs:
        if "Meeting Start Time" in s.get_text():
            full_text = s.get_text(separator=" ", strip=True)
            date_part = full_text.split("(")[0].strip()
            races_part = full_text.split("races")[0].split("-")[-1].strip()
            time_part = full_text.split("Meeting Start Time:")[-1].strip()
            return date_part, races_part, time_part
    return None, None, None

def fetch_race_data(date_obj):
    base_url = "https://racing.hkjc.com/racing/english/racing-widget/meetingreminder.aspx?raceDate="
    date_str = date_obj.strftime("%Y%m%d")
    print(f"Fetching data for {date_str}...")
    url = base_url + date_str
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            date_text, races, start_time = extract_info_from_html(response.text)
            if date_text and races and start_time:
                return {
                    "date": date_text,
                    "race": races,
                    "start time": start_time
                }
    except Exception as e:
        print(f"{date_str} failed: {e}")
    return None

def scrape_race_start_times(start_date, end_date):
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(fetch_race_data, date_range):
            if result:
                results.append(result)
    return pd.DataFrame(results)
# %%
# Run the scraper
start_date = datetime.strptime("20100101", "%Y%m%d")
end_date = datetime.today()
df = scrape_race_start_times(start_date, end_date)
df.to_csv("hkjc_race_start_times.csv", index=False)
print("Done! Saved to hkjc_race_start_times.csv")

# %%
df = pd.read_csv("hkjc_race_start_times.csv")

# Read and process the race place data, Format columns
race_date_df = df
race_date_df = race_date_df[["Date", "Course", "RaceNumber"]].drop_duplicates()
race_date_df = race_date_df.dropna(subset=["RaceNumber"])
race_date_df["RaceNumber"] = race_date_df["RaceNumber"].astype(int)
race_date_df["Date"] = pd.to_datetime(race_date_df["Date"], format="%Y/%m/%d", errors='coerce')
race_date_df["Date"] = race_date_df["Date"].dt.strftime("%d/%m/%Y")
race_date_df.to_csv("race_date.csv", index=False)

# Read the start times data
df_start = pd.read_csv("hkjc_race_start_times.csv")
df_start['RaceNumber'] = int(1)
df_start['Date'] = pd.to_datetime(df_start['Date'], format="%d/%m/%Y", errors='coerce')
df_start['Date'] = df_start['Date'].dt.strftime("%d/%m/%Y")
