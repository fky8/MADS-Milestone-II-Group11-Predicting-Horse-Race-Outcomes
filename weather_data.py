# %%
import pandas as pd

# Read and process the race place data
race_date_df = pd.read_csv("RacePlaceData_2010_2025.csv")
race_date_df = race_date_df[["Date", "Course", "RaceNumber"]].drop_duplicates()
race_date_df.to_csv("race_date.csv", index=False)
# %%
# Format columns
race_date_df = race_date_df.dropna(subset=["RaceNumber"])
race_date_df["RaceNumber"] = race_date_df["RaceNumber"].astype(int)
race_date_df["Date"] = pd.to_datetime(race_date_df["Date"], format="%Y/%m/%d", errors='coerce')
race_date_df["Date"] = race_date_df["Date"].dt.strftime("%d/%m/%Y")

# Read the start times data
df_start = pd.read_csv("hkjc_race_start_times.csv")
df_start['RaceNumber'] = int(1)
df_start['Date'] = pd.to_datetime(df_start['Date'], format="%d/%m/%Y", errors='coerce')
df_start['Date'] = df_start['Date'].dt.strftime("%d/%m/%Y")

# Merge on date and race number
df_merged = pd.merge(
    race_date_df,
    df_start,
    left_on=['Date', 'RaceNumber'],
    right_on=['Date', 'RaceNumber'],
    how='left'
)

# Propagate TotalRaceNumber from RaceNumber == 1 to all rows on same Date
df_merged['TotalRaceNumber'] = df_merged.groupby('Date')['TotalRaceNumber'].transform('first')

# Drop duplicate join columns if desired
# df_merged = df_merged.drop(columns=['date', 'race'])


# After propagating TotalRaceNumber, compute ComputedStartTime for each race on a given date
from datetime import timedelta

# Combine Date and StartTime to create full datetime for race 1
df_merged["StartDateTime"] = pd.to_datetime(df_merged["Date"] + " " + df_merged["StartTime"], format="%d/%m/%Y %H:%M", errors='coerce')

# Group by Date and get the first race start time
first_race_times = df_merged[df_merged["RaceNumber"] == 1][["Date", "StartDateTime"]].dropna()
df_merged = pd.merge(df_merged, first_race_times, on="Date", suffixes=("", "_first"))

# Compute start time for each race by adding (RaceNumber - 1) * 35 minutes
df_merged["ComputedStartTime"] = df_merged["StartDateTime_first"] + pd.to_timedelta((df_merged["RaceNumber"] - 1) * 35, unit='m')

# Format ComputedStartTime as HH:MM
df_merged["ComputedStartTime"] = df_merged["ComputedStartTime"].dt.strftime("%H:%M")
df_merged=df_merged[['Date', 'Course', 'RaceNumber','ComputedStartTime']]

# Format column dtype
df_merged = df_merged.astype({'Course': str, 'RaceNumber': int})
df_merged['Date'] = pd.to_datetime(df_merged['Date'], format="%d/%m/%Y")
df_merged['ComputedStartTime'] = pd.to_datetime(df_merged['ComputedStartTime'], format="%H:%M").dt.time

# Save the final DataFrame to CSV
df_merged.to_csv("race_date.csv", index=False)
print("Done")
# %%
print(df_merged.dtypes)

# %%
