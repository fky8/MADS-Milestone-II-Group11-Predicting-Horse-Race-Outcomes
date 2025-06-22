import os
import pandas as pd

COMMENTS_CSV = os.path.join("data", "Race_comments_gear_horse_competitors_2019_2025.csv")
WEATHER_CSV = os.path.join("data", "weather_data", "race_date_with_weather.csv")
OUTPUT_CSV = os.path.join("data", "Race_comments_gear_horse_competitors_weather_2019_2025.csv")


def merge_comments_with_weather(
    comments_csv: str = COMMENTS_CSV,
    weather_csv: str = WEATHER_CSV,
    output_csv: str = OUTPUT_CSV,
) -> None:
    """
    Merge race comments and competitor data with weather data on Date and Course.
    Saves the merged DataFrame as a new CSV.
    """
    comments = pd.read_csv(comments_csv)
    weather = pd.read_csv(weather_csv)

    # Ensure Date columns are datetime for both DataFrames
    comments["Date"] = pd.to_datetime(comments["Date"], errors="coerce")
    weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")

    # Merge on Date and Course
    merged = comments.merge(weather, on=["Date", "Course"], how="left")
    merged.to_csv(output_csv, index=False)
    print(f"Saved {len(merged)} rows to {output_csv}")


if __name__ == "__main__":
    merge_comments_with_weather()
