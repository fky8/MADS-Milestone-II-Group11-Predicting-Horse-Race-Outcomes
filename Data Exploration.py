# %%
import pandas as pd

# %%
race_df = pd.read_csv("RacePlaceData_ALL_YEARS.csv")
horse_df = pd.read_csv("hkjc_horse_profiles.csv")
gear_df = pd.read_csv("comments_2010_to_2025_expanded.csv")
# Remove any unnamed columns that came from CSV export
race_df = race_df.loc[:, ~race_df.columns.str.startswith('Unnamed')]
horse_df = horse_df.loc[:, ~horse_df.columns.str.startswith('Unnamed')]
gear_df = gear_df.loc[:, ~gear_df.columns.str.startswith('Unnamed')]
# %%
# Data Cleaning: column alignment, index extraction, dtype & datetime parsing
# Standardize and unify column names across DataFrames
race_df.rename(columns={'Pla.': 'Placing'}, inplace=True)
gear_df.rename(columns={'Horse No': 'Horse No.', 'Horse Name': 'Horse'}, inplace=True)
horse_df.drop(columns=['Letter', 'ProfileURL'], inplace=True)

horse_df.rename(columns={'HorseName': 'Horse'}, inplace=True)

 # Extract Horse index and clean Horse name in race_df
race_df['HorseIndex'] = race_df['Horse'].str.extract(r'\(([^)]+)\)')
race_df['Horse'] = race_df['Horse'].str.replace(r'\s*\([^)]*\)', '', regex=True)

 # Extract Horse index and clean Horse name in gear_df
gear_df['HorseIndex'] = gear_df['Horse'].str.extract(r'\(([^)]+)\)')
gear_df['Horse'] = gear_df['Horse'].str.replace(r'\s*\([^)]*\)', '', regex=True)

 # Ensure merge keys have matching dtypes
# Convert RaceNumber and Horse No. to pandas nullable integer type for both DataFrames
race_df['RaceNumber'] = race_df['RaceNumber'].astype('Int64')
gear_df['RaceNumber'] = gear_df['RaceNumber'].astype('Int64')
race_df['Horse No.'] = race_df['Horse No.'].astype('Int64')
gear_df['Horse No.'] = gear_df['Horse No.'].astype('Int64')


# Preserve original date strings for later diagnosis
gear_df['Date_str'] = gear_df['Date'].astype(str)
race_df['Date_str'] = race_df['Date'].astype(str)



# Parse race_df dates handling both D/M/YYYY and YYYY/MM/DD with slashes
date_series = race_df['Date_str']
# First parse day/month/year
race_df['Date'] = pd.to_datetime(date_series, format='%d/%m/%Y', errors='coerce')
# Then fill in ISO year/month/day for remaining
mask = race_df['Date'].isna()
race_df.loc[mask, 'Date'] = pd.to_datetime(
    date_series[mask], format='%Y/%m/%d', errors='coerce'
)

# Parse gear_df dates in ISO year-month-day format
gear_df['Date'] = pd.to_datetime(
    gear_df['Date_str'], format='%Y-%m-%d', errors='coerce'
)


# Extract unparsable date entries where coercion produced NaT
unparsable_gear = gear_df[gear_df['Date'].isna()]['Date_str']
unparsable_race = race_df[race_df['Date'].isna()]['Date_str']
print("Unparsable gear dates:")
print(unparsable_gear.drop_duplicates())
print("Unparsable race dates:")
print(unparsable_race.drop_duplicates())


# %%
# %%
# %%
# Combine race and gear DataFrames by coalescing on keys
keys = ['Date', 'RaceNumber', 'Horse No.']
race_gear_df = (
    race_df.set_index(keys)
    .combine_first(gear_df.set_index(keys))
    .reset_index()
)
print (race_gear_df.columns)
# Reorder columns: race_df columns first, then gear-only columns
race_cols = list(race_df.columns)
gear_only_cols = [col for col in race_gear_df.columns if col in gear_df.columns and col not in race_cols]
race_gear_df = race_gear_df[race_cols + gear_only_cols]
print("Reordered race_gear_df columns:", race_gear_df.columns)
# race_gear_df.to_csv("race_gear.csv", index=False)
# %%
print (horse_df.columns)
# %%
for df in (race_df, gear_df):
    print(df.dtypes[['Date','RaceNumber','Horse No.']])

common_idx = (race_df.set_index(keys)
                    .index.intersection(gear_df.set_index(keys).index))
print(f"Matching rows: {len(common_idx)}")
# %%
# Combine race_gear_df and horse_df by coalescing on Horse (approach 4)
horse_profile = horse_df.set_index('Horse')
race_full_df = (
    race_gear_df.set_index('Horse')
    .combine_first(horse_profile)
    .reset_index()
)
print(race_full_df.columns)
race_full_df.to_csv("Race_comments_gear_expanded.csv", index=False)

# %%
print(race_full_df.head(20))
# %%
pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(race_full_df.columns)
# %%
