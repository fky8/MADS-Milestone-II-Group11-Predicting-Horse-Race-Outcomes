# %%
import pandas as pd
import re
import numpy as np

# %%
race_df = pd.read_csv("RacePlaceData_2010_2025.csv")
horse_df = pd.read_csv("hkjc_horse_profiles.csv")
gear_df = pd.read_csv("comments_2010_to_2025_combined.csv")
# Remove any unnamed columns that came from CSV export
race_df = race_df.loc[:, ~race_df.columns.str.startswith('Unnamed')]
horse_df = horse_df.loc[:, ~horse_df.columns.str.startswith('Unnamed')]
gear_df = gear_df.loc[:, ~gear_df.columns.str.startswith('Unnamed')]
# %%
# Data Cleaning: column alignment, index extraction, dtype & datetime parsing
# Standardize and unify column names across DataFrames
race_df.rename(columns={'Pla.': 'Placing'}, inplace=True)
gear_df.rename(columns={'Horse No': 'Horse No.', 'Horse Name': 'Horse'}, inplace=True)
# horse_df.drop(columns=['Letter', 'ProfileURL'], inplace=True)

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

# Replace problematic date strings with pd.NA
race_df['Date_str'] = race_df['Date_str'].replace(['nan', 'NaN', ''], pd.NA)
gear_df['Date_str'] = gear_df['Date_str'].replace(['nan', 'NaN', ''], pd.NA)


# Parse race_df dates handling both D/M/YYYY and YYYY/MM/DD with slashes
date_series = race_df['Date_str']
# First parse day/month/year
race_df['Date'] = pd.to_datetime(date_series, format='%Y/%m/%d', errors='coerce')
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
print(race_df['Date'].iloc[1].year)
print(gear_df['Date'].iloc[1].day)
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
# Print all columns and their data types for race_full_df
print("race_full_df columns and dtypes:")
print(race_full_df.dtypes)

# Reorder columns in race_full_df: race_df columns, then horse_df columns excluding overlaps, then gear_df columns excluding overlaps
race_cols_set = set(race_df.columns)
horse_cols = [col for col in horse_df.columns if col not in race_cols_set and col != 'Horse']
gear_cols_set = set(gear_df.columns)
race_gear_cols_set = set(race_gear_df.columns)
gear_only_cols = [col for col in gear_df.columns if col not in race_cols_set and col not in horse_cols and col != 'Horse']
ordered_cols = list(race_df.columns) + horse_cols + gear_only_cols
race_full_df = race_full_df[ordered_cols]

# Sort race_full_df by Date, RaceNumber, and Placing
race_full_df = race_full_df.sort_values(by=['Date', 'RaceNumber', 'Placing'])

# Remove all dollar signs from string values in the DataFrame, excluding column names
def remove_dollar_signs(val):
    if isinstance(val, str):
        return val.replace('$', '')
    return val
race_full_df = race_full_df.applymap(remove_dollar_signs)

# Drop rows where 'Horse' or 'Placing' is missing
race_full_df = race_full_df.dropna(subset=['Horse', 'Placing'])

# Drop completely blank rows (all values are NaN)
race_full_df = race_full_df.dropna(how='all')

# race_full_df.to_csv("Race_comments_gear_ordered.csv", index=False)

# Split into two files based on date range
# mask_2010_2018 = (race_full_df['Date'] >= '2010-01-01') & (race_full_df['Date'] <= '2018-12-31')
# mask_2019_2025 = (race_full_df['Date'] >= '2019-01-01') & (race_full_df['Date'] <= '2025-12-31')

# race_full_df.loc[mask_2010_2018].to_csv("Race_comments_gear_ordered_2010_2018.csv", index=False)
# race_full_df.loc[mask_2019_2025].to_csv("Race_comments_gear_ordered_2019_2025.csv", index=False)

# %%
race_date_df = pd.read_csv("RacePlaceData_2010_2025.csv").drop_duplicates()
race_date_df = race_date_df[['Date', 'Course', 'RaceNumber']]
race_date_df.to_csv("race_date.csv", index=False)
# %%
# Add HorseCompetitor1 to HorseCompetitor13 columns
def assign_competitors(group):
    horse_indices = group['HorseIndex'].tolist()
    for idx, row in group.iterrows():
        # Exclude the subject horse from the competitors
        competitors = [h for h in horse_indices if h != row['HorseIndex']]
        # Pad with None if fewer than 13 competitors
        for i in range(1, 14):
            col_name = f'HorseCompetitor{i}'
            group.at[idx, col_name] = competitors[i-1] if i-1 < len(competitors) else None
    return group

race_full_df = race_full_df.groupby(['Date', 'RaceNumber'], group_keys=False).apply(assign_competitors)

# Sort columns to move new competitor columns to the end
competitor_cols = [col for col in race_full_df.columns if 'HorseCompetitor' in col]
other_cols = [col for col in race_full_df.columns if 'HorseCompetitor' not in col]
race_full_df = race_full_df[other_cols + competitor_cols]

# Add JockeyCompetitor1 to JockeyCompetitor13 columns
def assign_jockeys(group):
    jockeys = group['Jockey'].tolist()
    for idx, row in group.iterrows():
        competitors = [j for j in jockeys if j != row['Jockey']]
        for i in range(1, 14):
            col_name = f'JockeyCompetitor{i}'
            group.at[idx, col_name] = competitors[i-1] if i-1 < len(competitors) else None
    return group

race_full_df = race_full_df.groupby(['Date', 'RaceNumber'], group_keys=False).apply(assign_jockeys)

# Sort columns to move new competitor columns to the end
jockey_cols = [col for col in race_full_df.columns if 'JockeyCompetitor' in col]
other_cols = [col for col in race_full_df.columns if 'JockeyCompetitor' not in col]
race_full_df = race_full_df[other_cols + jockey_cols]



def clean_lbw(val):
    if pd.isna(val) or str(val).strip() == '':
        return float('nan')
    val = str(val).strip()
    if val == '-':
        return 0.0
    # Match patterns like "20-1/4" or "2-3/4"
    match = re.match(r'^(\d+)-(\d+)/(\d+)$', val)
    if match:
        whole = int(match.group(1))
        numerator = int(match.group(2))
        denominator = int(match.group(3))
        return whole + numerator / denominator
    # Match patterns like "1/4"
    match = re.match(r'^(\d+)/(\d+)$', val)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        return numerator / denominator
    # Try to convert directly to float
    try:
        return float(val)
    except Exception:
        return float('nan')

if 'LBW' in race_full_df.columns:
    race_full_df['LBW'] = race_full_df['LBW'].apply(clean_lbw).astype(float)

# Clean and convert Distance column
if 'Distance' in race_full_df.columns:
    race_full_df = race_full_df.rename(columns={'Distance': 'DistanceMeter'})
    # Remove 'M', strip spaces, handle NaN, and convert to integer
    race_full_df['DistanceMeter'] = (
        race_full_df['DistanceMeter']
        .astype(str)
        .str.replace('M', '', regex=False)
        .str.strip()
    )
    # Convert to numeric, coercing errors to NaN, then to Int64 for nullable integer
    race_full_df['DistanceMeter'] = pd.to_numeric(race_full_df['DistanceMeter'], errors='coerce').astype('Int64')

def split_score_range(val):
    if pd.isna(val) or str(val).strip() == '':
        return pd.Series({'MinScore': np.nan, 'MaxScore': np.nan})
    val = str(val).strip()
    # Pattern: "80-60" or "95-80"
    match = re.match(r'^(\d+)\s*-\s*(\d+)$', val)
    if match:
        max_score = int(match.group(1))
        min_score = int(match.group(2))
        return pd.Series({'MinScore': min_score, 'MaxScore': max_score})
    # Pattern: "95+"
    match = re.match(r'^(\d+)\+$', val)
    if match:
        min_score = int(match.group(1))
        return pd.Series({'MinScore': min_score, 'MaxScore': np.nan})
    # Pattern: just a number
    match = re.match(r'^(\d+)$', val)
    if match:
        score = int(match.group(1))
        return pd.Series({'MinScore': score, 'MaxScore': score})
    return pd.Series({'MinScore': np.nan, 'MaxScore': np.nan})

if 'Score range' in race_full_df.columns:
    score_split = race_full_df['Score range'].apply(split_score_range)
    # Convert to Int64 for nullable integer
    score_split['MinScore'] = pd.to_numeric(score_split['MinScore'], errors='coerce').astype('Int64')
    score_split['MaxScore'] = pd.to_numeric(score_split['MaxScore'], errors='coerce').astype('Int64')
    race_full_df = pd.concat([race_full_df, score_split], axis=1)

# final part of code:
mask_2010_2018 = (race_full_df['Date'] >= '2010-01-01') & (race_full_df['Date'] <= '2018-12-31')
mask_2019_2025 = (race_full_df['Date'] >= '2019-01-01') & (race_full_df['Date'] <= '2025-12-31')

race_full_df.loc[mask_2010_2018].to_csv("Race_comments_gear_ordered_with_competitors_2010_2018.csv", index=False)
race_full_df.loc[mask_2019_2025].to_csv("Race_comments_gear_ordered_with_competitors_2019_2025.csv", index=False)
