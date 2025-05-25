import pandas as pd
df_start = pd.read_csv("hkjc_race_start_times.csv")
df_start["RaceNumber"] = 1
# %%
race_date_df = pd.read_csv("race_date.csv")
race_date_df = race_date_df.dropna(subset=["RaceNumber"])
race_date_df["RaceNumber"] = race_date_df["RaceNumber"].astype(int)
race_date_df["Date"] = pd.to_datetime(race_date_df["Date"], format="%Y/%m/%d", errors='coerce')
race_date_df["Date"] = race_date_df["Date"].dt.strftime("%d/%m/%Y")

# %%
# Merge on date and race number
df_merged = pd.merge(
    race_date_df,
    df_start,
    on=['Date', 'RaceNumber'],
    how='left'
)

# Propagate TotalRaceNumber from RaceNumber == 1 to all rows on same Date
df_merged['TotalRaceNumber'] = df_merged.groupby('Date')['TotalRaceNumber'].transform('first')


# After propagating TotalRaceNumber, compute ComputedStartTime for each race on a given date
from datetime import timedelta

# Combine Date and StartTime to create full datetime for race 1
df_merged["StartDateTime"] = pd.to_datetime(df_merged["Date"] + " " + df_merged["StartTime"], format="%d/%m/%Y %H:%M", errors='coerce')

# Group by Date and get the first race start time
first_race_times = df_merged[df_merged["RaceNumber"] == 1][["Date", "StartDateTime"]].dropna()
df_merged = pd.merge(df_merged, first_race_times, on="Date", suffixes=("", "_first"))

# Compute start time for each race by adding (RaceNumber - 1) * 35 minutes
df_merged.loc[df_merged['StartTime'].isna(), 'ComputedStartTime'] = (
    df_merged['StartDateTime_first'] + pd.to_timedelta((df_merged["RaceNumber"] - 1) * 35, unit='m')
).dt.strftime("%H:%M")
df_merged.loc[df_merged['StartTime'].notna(), 'ComputedStartTime'] = df_merged.loc[df_merged['StartTime'].notna(), 'StartTime']

df_merged=df_merged[['Date', 'Course', 'RaceNumber','ComputedStartTime']]

# Format column dtype
df_merged = df_merged.astype({'Course': str, 'RaceNumber': int})
df_merged['Date'] = pd.to_datetime(df_merged['Date'], format="%d/%m/%Y")
df_merged['ComputedStartTime'] = pd.to_datetime(df_merged['ComputedStartTime'], format="%H:%M", errors='coerce').dt.time

# Save the final DataFrame to CSV
df_merged.to_csv("Race_date_with_computed_start_time.csv", index=False)
print("Done")

# # Read input files
# df_times = pd.read_csv("Race_date_with_computed_start_time.csv", parse_dates=["Date"])
# df_comments = pd.read_csv("Race_comments_gear_ordered.csv", low_memory=False)

# # Ensure dtypes match for merge keys
# df_times["RaceNumber"] = df_times["RaceNumber"].astype(int)
# df_comments["Date"] = pd.to_datetime(df_comments["Date"], format="%Y/%m/%d", errors='coerce')
# df_comments["RaceNumber"] = df_comments["RaceNumber"].astype(int)

# # Merge computed start time
# df_comments = df_comments.merge(
#     df_times,
#     on=["Date", "RaceNumber"],
#     how="left"
# )

# # Save result
# df_comments.to_csv("Race_comments_gear_ordered_with_start_time.csv", index=False)
# # %%
# print (df_times[["Date","RaceNumber"]].dtypes)
# print (df_comments[["Date","RaceNumber"]].dtypes)
# # %%
