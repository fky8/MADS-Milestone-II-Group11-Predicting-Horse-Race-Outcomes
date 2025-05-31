# %%
import pandas as pd
import requests
from datetime import datetime, timedelta
# Load race data
df = pd.read_csv('race_date.csv').drop(["RaceNumber"],axis=1).drop_duplicates()
# Add CourseName column with human-readable names
df['CourseName'] = df['Course'].map({'ST': 'ShaTin', 'HV': 'HappyValley'})
print (df.head(5))

# %%
venue_to_station = {
    'ST': 'SHA',
    'HV': 'HPV'
}
station_year_cache = {}

# def fetch_weather_for_race(race_date, venue):
#     # Construct URL with station and year parameters
#     url = (
#         f'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php'
#         f'?dataType=CLMTEMP&lang=en&rformat=json&station={venue}&year={race_date.year}'
#     )

#     try:
#         response = requests.get(url)
#         data = response.json()
#         print(data)  # print the json, select keys that match ["2000","1","2","19.9","C"] 
#     except Exception as e:
#         print(f"Error fetching weather for {race_date}: {e}")
#         return {}

# Add columns for Mean, Max and Min Temperature
df['MeanTemperature'] = None
df['MaxTemperature'] = None
df['MinTemperature'] = None

# Mapping of datatype to column name
datatype_column_map = {
    'CLMTEMP': 'MeanTemperature',
    'CLMMAXT': 'MaxTemperature',
    'CLMMINT': 'MinTemperature'
}

# Cache all responses by (station, year, datatype)
station_year_cache = {}

# Loop through each row of df, fetch weather and populate temperature columns
for idx, row in df.iterrows():
    race_date = pd.to_datetime(row['Date'], errors='coerce')
    if pd.isna(race_date):
        print(f"Skipping row {idx}: invalid or missing date '{row['Date']}'")
        continue
    venue = row['Course']
    station = venue_to_station.get(venue, venue)

    for datatype, column_name in datatype_column_map.items():
        cache_key = (station, race_date.year, datatype)
        if cache_key not in station_year_cache:
            url = (
                f'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php'
                f'?dataType={datatype}&lang=en&rformat=json&station={station}&year={race_date.year}'
            )
            try:
                response = requests.get(url)
                station_year_cache[cache_key] = response.json()
            except Exception as e:
                print(f"Error fetching {datatype} for {race_date}: {e}")
                station_year_cache[cache_key] = {}

        data = station_year_cache[cache_key]
        try:
            for entry in data.get("data", []):
                year, month, day = entry[:3]
                if (int(year) == race_date.year and int(month) == race_date.month and int(day) == race_date.day):
                    df.at[idx, column_name] = entry[3]
                    print(f"Updated {column_name}:", race_date, entry[3])
                    break
        except Exception as e:
            print(f"Error processing {datatype} data for {race_date}: {e}")

    # Fetch RYES only for dates on or after 2019-09-10
    if race_date >= datetime(2019, 9, 10):
        date_str = race_date.strftime('%Y%m%d')
        ryes_url = (
            f'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php'
            f'?dataType=RYES&lang=en&rformat=json&date={date_str}'
        )
        try:
            ryes_response = requests.get(ryes_url)
            ryes_data = ryes_response.json()
            # Extract desired metrics if present
            for metric in ['HKOReadingsMaxRH', 'HKOReadingsMinRH',
                           'HKOReadingsMinGrassTemp', 'HKOReadingsRainfall',
                           'KingsParkReadingsMeanUVIndex', 'KingsParkReadingsMaxUVIndex']:
                if metric in ryes_data:
                    df.at[idx, metric] = ryes_data[metric]
        except Exception as e:
            print(f"Error fetching RYES data for {race_date}: {e}")
    else:
        # Skip RYES lookup for dates before threshold
        print(f"Skipping RYES for {race_date.strftime('%Y-%m-%d')} (before 2019-09-10)")


# %%
# Rename columns before saving
df = df.rename(columns={
    'HKOReadingsMaxRH': 'MaximumRelativeHumidity',
    'HKOReadingsMinRH': 'MinimumRelativeHumidity',
    'HKOReadingsMinGrassTemp': 'GrassMinimumTemperature',
    'HKOReadingsRainfall': 'Rainfall(mm)',
    'KingsParkReadingsMeanUVIndex': 'MeanUVIndex',
    'KingsParkReadingsMaxUVIndex': 'MaxUVIndex'
})

# %%
# Replace 'Trace' with 0 in Rainfall(mm) column before saving
if 'Rainfall(mm)' in df.columns:
    df['Rainfall(mm)'] = df['Rainfall(mm)'].replace(['Trace', 'trace'], 0)
    df['Rainfall(mm)'] = pd.to_numeric(df['Rainfall(mm)'], errors='coerce')
df.to_csv('race_date_with_weather.csv', index=False)

# %%
print (df.columns)
# %%
