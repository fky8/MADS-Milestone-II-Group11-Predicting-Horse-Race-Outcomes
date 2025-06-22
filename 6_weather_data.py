import os
import pandas as pd
from datetime import timedelta


def load_data(
    start_csv: str = os.path.join("data", "race", "hkjc_race_start_times.csv"),
    race_csv: str = os.path.join("data", "race", "race_date.csv"),
):
    """
    Load and align race start time and race date data.
    Returns two DataFrames: start times and race dates.
    """
    df_start = pd.read_csv(start_csv)
    df_start = df_start.rename(
        columns={
            "date": "Date",
            "race": "TotalRaceNumber",
            "start time": "StartTime",
        }
    )
    df_start["RaceNumber"] = 1

    race_date_df = pd.read_csv(race_csv)
    race_date_df = race_date_df.dropna(subset=["RaceNumber"])
    race_date_df["RaceNumber"] = race_date_df["RaceNumber"].astype(int)
    race_date_df["Date"] = pd.to_datetime(
        race_date_df["Date"], format="%d/%m/%Y", errors="coerce"
    ).dt.strftime("%d/%m/%Y")
    return df_start, race_date_df


def merge_start_times(df_start: pd.DataFrame, race_date_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge start times with race data and compute per-race times.
    If a race's start time is missing, estimate it by adding 35 minutes per race.
    """
    df = pd.merge(race_date_df, df_start, on=["Date", "RaceNumber"], how="left")
    df["TotalRaceNumber"] = df.groupby("Date")["TotalRaceNumber"].transform("first")

    df["StartDateTime"] = pd.to_datetime(
        df["Date"] + " " + df["StartTime"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    first_race = df[df["RaceNumber"] == 1][["Date", "StartDateTime"]].dropna()
    df = df.merge(first_race, on="Date", suffixes=("", "_first"))

    # If StartTime is missing, estimate using first race's time + 35min per race
    df.loc[df["StartTime"].isna(), "ComputedStartTime"] = (
        df["StartDateTime_first"] + pd.to_timedelta((df["RaceNumber"] - 1) * 35, unit="m")
    ).dt.strftime("%H:%M")
    df.loc[df["StartTime"].notna(), "ComputedStartTime"] = df.loc[
        df["StartTime"].notna(), "StartTime"
    ]

    df = df[["Date", "Course", "RaceNumber", "ComputedStartTime"]]
    df = df.astype({"Course": str, "RaceNumber": int})
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["ComputedStartTime"] = pd.to_datetime(
        df["ComputedStartTime"], format="%H:%M", errors="coerce"
    ).dt.time
    return df


def main() -> None:
    """
    Main workflow:
    1. Load start time and race date data.
    2. Merge and compute per-race start times.
    3. Save the merged result.
    """
    df_start, race_date_df = load_data()
    merged = merge_start_times(df_start, race_date_df)
    merged.to_csv("data/race/Race_date_with_computed_start_time.csv", index=False)
    print("Done")


if __name__ == "__main__":
    main()
