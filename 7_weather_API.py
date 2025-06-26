import os
import pandas as pd
import requests
from datetime import datetime


def fetch_weather(
    race_csv: str = os.path.join("data", "race", "race_date.csv"),
    output_csv: str = os.path.join("data", "weather_data", "race_date_with_weather.csv"),
) -> None:
    """
    Fetch daily weather data for each race date and merge with race info.
    - Uses HK weather API for temperature and humidity.
    - Adds RYES data for races after 2019-09-10.
    """
    df = pd.read_csv(race_csv).drop(["RaceNumber"], axis=1).drop_duplicates()
    df["CourseName"] = df["Course"].map({"ST": "ShaTin", "HV": "HappyValley"})

    venue_to_station = {"ST": "SHA", "HV": "HPV"}
    datatype_column_map = {
        "CLMTEMP": "MeanTemperature",
        "CLMMAXT": "MaxTemperature",
        "CLMMINT": "MinTemperature",
    }
    station_year_cache: dict[tuple, dict] = {}
    df["MeanTemperature"] = None
    df["MaxTemperature"] = None
    df["MinTemperature"] = None

    for idx, row in df.iterrows():
        race_date = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(race_date):
            print(f"Skipping row {idx}: invalid or missing date '{row['Date']}'")
            continue
        station = venue_to_station.get(row["Course"], row["Course"])

        # Fetch and cache temperature data for each station/year/datatype
        for datatype, column_name in datatype_column_map.items():
            cache_key = (station, race_date.year, datatype)
            if cache_key not in station_year_cache:
                url = (
                    "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
                    f"?dataType={datatype}&lang=en&rformat=json&station={station}&year={race_date.year}"
                )
                try:
                    response = requests.get(url)
                    station_year_cache[cache_key] = response.json()
                except Exception as exc:
                    print(f"Error fetching {datatype} for {race_date}: {exc}")
                    station_year_cache[cache_key] = {}

            data = station_year_cache[cache_key]
            try:
                for entry in data.get("data", []):
                    year, month, day = entry[:3]
                    if int(year) == race_date.year and int(month) == race_date.month and int(day) == race_date.day:
                        df.at[idx, column_name] = entry[3]
                        break
            except Exception as exc:
                print(f"Error processing {datatype} data for {race_date}: {exc}")

        # Fetch RYES data for races after 2019-09-10
        if race_date >= datetime(2019, 9, 10):
            date_str = race_date.strftime("%Y%m%d")
            ryes_url = (
                "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
                f"?dataType=RYES&lang=en&rformat=json&date={date_str}"
            )
            try:
                ryes_response = requests.get(ryes_url)
                ryes_data = ryes_response.json()
                for metric in [
                    "HKOReadingsMaxRH",
                    "HKOReadingsMinRH",
                    "HKOReadingsMinGrassTemp",
                    "HKOReadingsRainfall",
                    "KingsParkReadingsMeanUVIndex",
                    "KingsParkReadingsMaxUVIndex",
                ]:
                    if metric in ryes_data:
                        df.at[idx, metric] = ryes_data[metric]
            except Exception as exc:
                print(f"Error fetching RYES data for {race_date}: {exc}")
        else:
            print(f"Skipping RYES for {race_date.strftime('%Y-%m-%d')} (before 2019-09-10)")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "HKOReadingsMaxRH": "MaximumRelativeHumidity",
            "HKOReadingsMinRH": "MinimumRelativeHumidity",
            "HKOReadingsMinGrassTemp": "GrassMinimumTemperature",
            "HKOReadingsRainfall": "Rainfall(mm)",
            "KingsParkReadingsMeanUVIndex": "MeanUVIndex",
            "KingsParkReadingsMaxUVIndex": "MaxUVIndex",
        }
    )

    # Clean up Rainfall column
    if "Rainfall(mm)" in df.columns:
        df["Rainfall(mm)"] = df["Rainfall(mm)"].replace(["Trace", "trace"], 0)
        df["Rainfall(mm)"] = pd.to_numeric(df["Rainfall(mm)"], errors="coerce")

    df.to_csv(output_csv, index=False)
    print(df.columns)


if __name__ == "__main__":
    fetch_weather()
