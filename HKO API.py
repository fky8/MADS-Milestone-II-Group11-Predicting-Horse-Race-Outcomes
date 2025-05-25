import pandas as pd
import requests
from datetime import datetime, timedelta

# Load race data
df = pd.read_csv('Race_date_with_computed_start_time.csv')
print("Columns in DataFrame:", df.columns.tolist())
df.columns = df.columns.str.strip()
print("Sample ComputedStartTime values:")
print(df['ComputedStartTime'].head(10))
df['ComputedStartTime'] = df['ComputedStartTime'].str.strip()
df['Start_time'] = pd.to_datetime(df['ComputedStartTime'], errors='coerce').dt.time
invalid_times = df[df['Start_time'].isna()]
print("Invalid ComputedStartTime entries:")
print(invalid_times[['ComputedStartTime']])
df['Race_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Start_time'].astype(str))

# Venue to HKO station map
venue_to_station = {
    'ST': 'Sha Tin',
    'HV': 'Happy Valley'
}

def fetch_weather_for_race(race_datetime, venue):
    date_str = race_datetime.strftime('%Y%m%d')
    url = f'https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=rhrread&lang=en'
    
    try:
        response = requests.get(url)
        data = response.json()

        station_name = venue_to_station.get(venue, None)
        if not station_name:
            return {}

        weather_obs = data.get("temperature", {}).get("data", [])
        temp = next((x["value"] for x in weather_obs if x["place"] == station_name), None)

        rh_obs = data.get("humidity", {}).get("data", [])
        rh = next((x["value"] for x in rh_obs if x["place"] == station_name), None)

        rainfall_obs = data.get("rainfall", {}).get("data", [])
        rain = next((x["max"] for x in rainfall_obs if x["place"] == station_name), None)

        return {
            'temperature': temp,
            'humidity': rh,
            'rainfall': rain
        }
    except Exception as e:
        print(f"Error fetching weather for {race_datetime}: {e}")
        return {}

# Loop through each race and append weather
weather_data = []

for i, row in df.iterrows():
    race_datetime = row['Race_datetime']
    venue = row['Course']
    weather = fetch_weather_for_race(race_datetime, venue)
    weather_data.append(weather)

weather_df = pd.DataFrame(weather_data)
df_with_weather = pd.concat([df, weather_df], axis=1)

# Save to new file
df_with_weather.to_csv('Race_with_weather.csv', index=False)